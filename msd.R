library(yardstick)
library(readr)

our_measures <- read_csv("mestrado/measure_project/output/our_measures.csv")
grand_mean <- read_csv("mestrado/measure_project/output/grand_mean.csv")

yardstick::msd_vec(
  as.vector(our_measures['neck_girth']),
  as.numeric(grand_mean['neck_girth'])
)