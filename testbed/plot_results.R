library("ggplot2")

results <- read.csv("testbed/results.tsv", sep = "\t")

# Plot:
# * Runtimes
# * Iterations
# * Memory usage
# * Error
# * Runtime vs. error
ggplot(results, aes(x = Algorithm, y = AverageRunTimeInMilliseconds, fill = Algorithm)) +
  geom_bar() +
  opts(legend.position = "none")
ggsave("testbed/graphs/run_times.pdf")

ggplot(results, aes(x = Algorithm, y = Iterations, fill = Algorithm)) +
  geom_bar() +
  opts(legend.position = "none")
ggsave("testbed/graphs/iterations.pdf")

ggplot(results, aes(x = reorder(Algorithm, Error), y = log1p(Error), fill = Algorithm)) +
  geom_bar() +
  scale_y_log10() +
  opts(legend.position = "none")
ggsave("testbed/graphs/solution_error.pdf")

ggplot(results, aes(x = AverageRunTimeInMilliseconds, y = Error, color = Algorithm)) +
  geom_point() +
  scale_y_log10()
ggsave("testbed/graphs/solution_error_vs_runtime.pdf")
