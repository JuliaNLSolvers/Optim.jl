library("ggplot2")

results <- read.csv("testbed/results.tsv", sep = "\t")

# Plot:
# * Runtimes
# * Iterations
# * Memory usage
# * Error
# * Runtime vs. error
ggplot(results, aes(x = Algorithm, y = log1p(AverageRunTimeInMilliseconds), fill = Algorithm)) +
  geom_bar() +
  coord_flip() +
  opts(legend.position = "none") +
  xlab("") +
  ylab("Average Run-time on a Log Scale") +
  opts(title = "Speed of Optimization Algorithms")
ggsave("testbed/graphs/run_times.pdf", width = 10, height = 9)

ggplot(results, aes(x = Algorithm, y = log1p(Iterations), fill = Algorithm)) +
  geom_bar() +
  coord_flip() +
  opts(legend.position = "none") +
  xlab("") +
  ylab("Iterations Used on a Log Scale") +
  opts(title = "Efficiency of Optimization Algorithms")
ggsave("testbed/graphs/iterations.pdf", width = 10, height = 9)

ggplot(results, aes(x = reorder(Algorithm, Error), y = log1p(Error + .Machine$double.eps)^(1/10), fill = Algorithm)) +
  geom_bar() +
  coord_flip() +
  opts(legend.position = "none") +
  xlab("") +
  ylab("Euclidean Norm of Error on a Root Log Scale") +
  opts(title = "Size of Errors in Solution from Optimization Algorithms")
ggsave("testbed/graphs/solution_error.pdf", width = 10, height = 9)

ggplot(results, aes(x = log1p(AverageRunTimeInMilliseconds), y = Error, color = Algorithm)) +
  geom_point() +
  scale_x_log10() +
  scale_y_log10() +
  #ylim(-.Machine$double.eps, max(results$Error)) +
  xlab("Average Run-time on a Log Scale") +
  ylab("Euclidean Norm of Error") +
  opts(title = "Speed vs. Errors in Solution from Optimization Algorithms")
ggsave("testbed/graphs/solution_error_vs_runtime.pdf", width = 12, height = 9)

# Also plot results without Nelder-Mead.
