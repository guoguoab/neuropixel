#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  required_pkgs <- c(
    "tidyverse", "tidymodels", "vip", "patchwork"
  )
})

install_if_missing <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, FUN.VALUE = logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    message("Installing missing packages: ", paste(missing, collapse = ", "))
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
}

install_if_missing(required_pkgs)

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(vip)
  library(patchwork)
})

tidymodels_prefer()

# ===============================
# 配置区域（可按需修改）
# ===============================
train_path <- "train.csv"
test_path <- "test.csv"
target_col <- "label"
positive_class <- NULL # 例如 "1"，NULL 时自动采用排序后的最后一个类别
seed <- 20260413
output_dir <- "model_outputs"

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
set.seed(seed)

# ===============================
# 数据读取与检查
# ===============================
if (!file.exists(train_path) || !file.exists(test_path)) {
  stop("未找到 train.csv 或 test.csv，请将数据文件放在脚本同级目录，或在脚本中修改 train_path/test_path。")
}

train_df <- readr::read_csv(train_path, show_col_types = FALSE)
test_df <- readr::read_csv(test_path, show_col_types = FALSE)

if (!(target_col %in% names(train_df)) || !(target_col %in% names(test_df))) {
  stop(sprintf("目标列 '%s' 不存在于 train/test 数据中。", target_col))
}

# 统一标签类型并处理类别
train_df <- train_df %>% mutate(!!target_col := as.factor(.data[[target_col]]))
test_df <- test_df %>% mutate(!!target_col := factor(.data[[target_col]], levels = levels(train_df[[target_col]])))

if (nlevels(train_df[[target_col]]) != 2) {
  stop("当前脚本针对二分类任务（2类标签）优化，请确保目标列只有2个类别。")
}

if (is.null(positive_class)) {
  positive_class <- levels(train_df[[target_col]])[2]
}

message("Positive class: ", positive_class)

# ===============================
# 配方：缺失值、类别变量、标准化
# ===============================
rec <- recipe(as.formula(paste(target_col, "~ .")), data = train_df) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_normalize(all_numeric_predictors())

# 使用分层交叉验证，提升稳定性
folds <- vfold_cv(train_df, v = 5, repeats = 2, strata = !!rlang::sym(target_col))

# ===============================
# 定义多模型（提升准确率候选）
# ===============================
model_specs <- list(
  logistic = logistic_reg(mode = "classification", penalty = tune(), mixture = tune()) %>%
    set_engine("glmnet"),

  rf = rand_forest(mode = "classification", trees = 600, mtry = tune(), min_n = tune()) %>%
    set_engine("ranger", importance = "impurity"),

  xgb = boost_tree(
    mode = "classification",
    trees = tune(),
    tree_depth = tune(),
    min_n = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    mtry = tune(),
    learn_rate = tune()
  ) %>% set_engine("xgboost"),

  svm = svm_rbf(mode = "classification", cost = tune(), rbf_sigma = tune()) %>%
    set_engine("kernlab")
)

make_wf <- function(spec) {
  workflow() %>%
    add_recipe(rec) %>%
    add_model(spec)
}

wfs <- purrr::map(model_specs, make_wf)

metric_set_used <- metric_set(accuracy, roc_auc, kap, sens, spec)
ctrl <- control_grid(save_pred = TRUE, save_workflow = TRUE, verbose = TRUE)

# 不同模型使用不同参数网格大小
grid_sizes <- c(logistic = 20, rf = 20, xgb = 30, svm = 20)

tune_one <- function(wf, model_name) {
  message("\n==== Tuning model: ", model_name, " ====")
  tune_grid(
    wf,
    resamples = folds,
    grid = grid_sizes[[model_name]],
    metrics = metric_set_used,
    control = ctrl
  )
}

tuned_results <- purrr::imap(wfs, tune_one)

# 汇总CV表现
cv_summary <- purrr::imap_dfr(tuned_results, function(x, nm) {
  collect_metrics(x) %>%
    filter(.metric %in% c("accuracy", "roc_auc")) %>%
    group_by(.metric) %>%
    slice_max(mean, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(model = nm)
})

readr::write_csv(cv_summary, file.path(output_dir, "cv_best_metrics.csv"))

# 选择最佳模型（优先ROC_AUC，其次Accuracy）
best_model_name <- cv_summary %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean)) %>%
  slice(1) %>%
  pull(model)

message("Best model by CV ROC_AUC: ", best_model_name)

best_tune <- tuned_results[[best_model_name]]
best_param <- select_best(best_tune, metric = "roc_auc")
final_wf <- finalize_workflow(wfs[[best_model_name]], best_param)

final_fit <- fit(final_wf, data = train_df)

# ===============================
# 测试集评估
# ===============================
pred_prob <- predict(final_fit, new_data = test_df, type = "prob")
pred_cls <- predict(final_fit, new_data = test_df, type = "class")

eval_df <- bind_cols(
  test_df %>% select(all_of(target_col)),
  pred_prob,
  pred_cls
)

# 统一正类概率列
prob_col <- paste0(".pred_", positive_class)
if (!(prob_col %in% names(eval_df))) {
  stop(sprintf("未找到正类概率列 %s，请确认标签命名。", prob_col))
}

test_metrics <- tibble(
  metric = c("accuracy", "roc_auc", "kap", "sens", "spec"),
  value = c(
    accuracy_vec(eval_df[[target_col]], eval_df$.pred_class),
    roc_auc_vec(eval_df[[target_col]], eval_df[[prob_col]], event_level = "second"),
    kap_vec(eval_df[[target_col]], eval_df$.pred_class),
    sens_vec(eval_df[[target_col]], eval_df$.pred_class, event_level = "second"),
    spec_vec(eval_df[[target_col]], eval_df$.pred_class, event_level = "second")
  )
)

readr::write_csv(test_metrics, file.path(output_dir, "test_metrics.csv"))

# 混淆矩阵
cm <- conf_mat(eval_df, truth = !!sym(target_col), estimate = .pred_class)
cm_tbl <- as_tibble(cm$table)
readr::write_csv(cm_tbl, file.path(output_dir, "confusion_matrix.csv"))

# ===============================
# 可视化（直观展示）
# ===============================
# 1) 各模型CV对比条形图
plot_cv <- cv_summary %>%
  select(model, .metric, mean) %>%
  mutate(.metric = factor(.metric, levels = c("roc_auc", "accuracy"))) %>%
  ggplot(aes(x = reorder(model, mean), y = mean, fill = .metric)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  coord_flip() +
  labs(
    title = "Cross-Validation Performance Comparison",
    x = "Model",
    y = "Mean Score",
    fill = "Metric"
  ) +
  theme_minimal(base_size = 12)

# 2) ROC曲线
roc_df <- yardstick::roc_curve(eval_df, truth = !!sym(target_col), !!sym(prob_col), event_level = "second")
auc_val <- roc_auc(eval_df, truth = !!sym(target_col), !!sym(prob_col), event_level = "second") %>% pull(.estimate)

plot_roc <- ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(size = 1.1, color = "#2C7FB8") +
  geom_abline(linetype = 2, color = "grey50") +
  coord_equal() +
  labs(
    title = sprintf("ROC Curve (Best: %s, AUC = %.4f)", best_model_name, auc_val),
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal(base_size = 12)

# 3) 混淆矩阵热力图
plot_cm <- cm_tbl %>%
  ggplot(aes(x = Prediction, y = Truth, fill = n)) +
  geom_tile(color = "white") +
  geom_text(aes(label = n), size = 5) +
  scale_fill_gradient(low = "#f7fbff", high = "#08306b") +
  labs(title = "Confusion Matrix (Test Set)", x = "Predicted", y = "Actual") +
  theme_minimal(base_size = 12)

# 4) 如果模型支持，展示特征重要性（如RF/XGB）
plot_vi <- NULL
if (best_model_name %in% c("rf", "xgb")) {
  plot_vi <- tryCatch({
    vi_plot <- vip::vip(final_fit$fit$fit$fit, num_features = 15) +
      labs(title = sprintf("Feature Importance (%s)", best_model_name))
    vi_plot
  }, error = function(e) {
    NULL
  })
}

if (is.null(plot_vi)) {
  final_plot <- plot_cv / (plot_roc | plot_cm)
} else {
  final_plot <- plot_cv / (plot_roc | plot_cm) / plot_vi
}

ggsave(
  filename = file.path(output_dir, "model_comparison_dashboard.png"),
  plot = final_plot,
  width = 14,
  height = 12,
  dpi = 300
)

ggsave(
  filename = file.path(output_dir, "cv_model_comparison.png"),
  plot = plot_cv,
  width = 9,
  height = 5,
  dpi = 300
)

ggsave(
  filename = file.path(output_dir, "roc_curve_test.png"),
  plot = plot_roc,
  width = 6,
  height = 6,
  dpi = 300
)

ggsave(
  filename = file.path(output_dir, "confusion_matrix_test.png"),
  plot = plot_cm,
  width = 6,
  height = 5,
  dpi = 300
)

# 保存预测结果
readr::write_csv(eval_df, file.path(output_dir, "test_predictions.csv"))

# 输出最终结论
message("\n======= Final Test Metrics =======")
print(test_metrics)
message("\n所有结果已保存到目录: ", normalizePath(output_dir))
message("包含: cv_best_metrics.csv, test_metrics.csv, confusion_matrix.csv, 以及多张PNG可视化图像。")
