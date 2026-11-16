data <- read.csv(
  "data/raw/student-mat.csv",
  sep = ";",
  header = TRUE
)

cat("Dataset Dimensions:\n")
print(dim(data))

cat("\nDescriptive Statistics:\n")
print(summary(data[, c("G1", "G2", "G3", "absences", "studytime")]))

cat("\nCorrelation Matrix:\n")
correlation <- cor(
  data[, c("G1", "G2", "G3", "absences", "studytime")],
  use = "complete.obs"
)
print(correlation)

hist(
  data$G3,
  main = "Final Grade Distribution",
  xlab = "Final Grade (G3)",
  col = "skyblue",
  border = "white"
)

boxplot(
  G3 ~ studytime,
  data = data,
  main = "Study Time vs Final Grade",
  xlab = "Study Time",
  ylab = "Final Grade",
  col = "lightgreen"
)

plot(
  data$absences,
  data$G3,
  main = "Absences vs Final Grade",
  xlab = "Absences",
  ylab = "Final Grade",
  pch = 19,
  col = "blue"
)

passed <- data$absences[data$G3 >= 10]
failed <- data$absences[data$G3 < 10]

cat("\nT-Test: Absences in Passed vs Failed Students\n")
print(t.test(passed, failed))

data$risk_level <- ifelse(
  data$G3 < 10, "High Risk",
  ifelse(data$G3 < 15, "Medium Risk", "Low Risk")
)

cat("\nRisk Distribution:\n")
print(table(data$risk_level))

cat("\nR Analysis Completed Successfully!\n")
