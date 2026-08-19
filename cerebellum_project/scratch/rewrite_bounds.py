import re

file_path = "src/R/manifold_extraction.R"
with open(file_path, "r") as f:
    content = f.read()

# Replace bounds
new_lower = "lower_bounds <- c(0.1, 0.3, 0.0, 0.0, -2.0, 0.1, 0.0, -2.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)"
new_upper = "upper_bounds <- c(4.0, 3.0, 0.4, 2.0, 2.0, 2.0, 10.0, 2.0, 2.0, 0.95, 0.50, 0.50, 1.0, 1.0, 1.0)"
new_initial = "initial_phi <- c(1.0, 1.5, 0.2, 0.5, 0.0, 0.5, 2.0, 0.0, 0.5, 0.5, 0.1, 0.05, 0.5, 0.5, 0.5)"

content = re.sub(r"lower_bounds <- c\(.*?\)", new_lower, content)
content = re.sub(r"upper_bounds <- c\(.*?\)", new_upper, content)
content = re.sub(r"initial_phi <- c\(.*?\)", new_initial, content)

with open(file_path, "w") as f:
    f.write(content)

file_path2 = "src/R/hierarchical_loocv_tournament.R"
with open(file_path2, "r") as f:
    content2 = f.read()

content2 = re.sub(r"lower_bounds <- c\(.*?\)", new_lower, content2)
content2 = re.sub(r"upper_bounds <- c\(.*?\)", new_upper, content2)

with open(file_path2, "w") as f:
    f.write(content2)

print("Replaced bounds in R scripts.")
