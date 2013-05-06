# Plot:
# * Runtimes
# * Iterations
# * Memory usage
# * Error
# * Runtime vs. error

# Also plot results without Nelder-Mead.

library("ggplot2")

results <- read.csv("benchmarks/results.tsv", sep = "\t")

n <- length(unique(results$Problem))

ggplot(results, aes(x = Algorithm, y = log1p(AverageRunTimeInMilliseconds), fill = Algorithm)) +
  geom_bar() +
  coord_flip() +
  theme(legend.position = "none") +
  xlab("") +
  ylab("Average Run-time on a Log Scale") +
  ggtitle("Speed of Optimization Algorithms") +
  facet_grid(Problem ~ .)
ggsave("benchmarks/graphs/run_times.png", width = 12, height = 9 * n)

ggplot(results, aes(x = Algorithm, y = log1p(Iterations), fill = Algorithm)) +
  geom_bar() +
  coord_flip() +
  theme(legend.position = "none") +
  xlab("") +
  ylab("Iterations Used on a Log Scale") +
  ggtitle("Efficiency of Optimization Algorithms") +
  facet_grid(Problem ~ .)
ggsave("benchmarks/graphs/iterations.png", width = 12, height = 9 * n)

ggplot(results, aes(x = reorder(Algorithm, Error), y = log1p(Error + .Machine$double.eps)^(1/10), fill = Algorithm)) +
  geom_bar() +
  coord_flip() +
  theme(legend.position = "none") +
  xlab("") +
  ylab("Euclidean Norm of Error on a Root Log Scale") +
  ggtitle("Size of Errors in Solution from Optimization Algorithms") +
  facet_grid(Problem ~ .)
ggsave("benchmarks/graphs/solution_error.png", width = 12, height = 9 * n)

ggplot(results, aes(x = log1p(AverageRunTimeInMilliseconds), y = Error, color = Algorithm)) +
  geom_point() +
  scale_x_log10() +
  scale_y_log10() +
  #ylim(-.Machine$double.eps, max(results$Error)) +
  xlab("Average Run-time on a Log Scale") +
  ylab("Euclidean Norm of Error") +
  ggtitle("Speed vs. Errors in Solution from Optimization Algorithms") +
  facet_grid(Problem ~ .)
ggsave("benchmarks/graphs/solution_error_vs_runtime.png", width = 12, height = 9 * n)
