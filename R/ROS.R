#' @title  Random Oversampling (ROS)
#'
#' @description Resampling with ROS.
#'
#' @param x feature matrix.
#' @param y a factor class variable with two classes.
#' @param ovRate Oversampling rate multiplied by the difference between maximum
#' and other of class sizes. Default is 1 meaning full balance.
#'
#' @details
#' Random Oversampling (ROS) is a method of copying and pasting of positive
#' samples until balance is achieved.
#'
#' Can work with classes more than 2.
#'
#' @return a list with resampled dataset.
#'  \item{x_new}{Resampled feature matrix.}
#'  \item{y_new}{Resampled target variable.}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @examples
#'
#' set.seed(1)
#' x <- rbind(matrix(rnorm(2000, 3, 1), ncol = 2, nrow = 1000),
#'            matrix(rnorm(100, 5, 1), ncol = 2, nrow = 50))
#' y <- as.factor(c(rep("negative", 1000), rep("positive", 50)))
#'
#' plot(x, col = y)
#'
#' # resampling
#' m <- ROS(x = x, y = y)
#'
#' plot(m$x_new, col = m$y_new)
#'
#' @rdname ROS
#' @export


ROS <- function(x, y, ovRate = 1) {

  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }

  if (!is.data.frame(x) & !is.matrix(x)) {
    stop("x must be a matrix or dataframe")
  }

  if (!is.factor(y)) {
    stop("y must be a factor")
  }

  var_names <- colnames(x)
  x <- as.matrix(x)
  p <- ncol(x)
  n <- nrow(x)

  class_names <- levels(y)
  n_classes <- sapply(class_names, function(m) sum(y == m))
  k_class <- length(class_names)
  n_classes_max <- max(n_classes)
  n_needed <- round((max(n_classes) - n_classes)*ovRate)

  x_classes <- lapply(class_names, function(m) x[y == m,, drop = FALSE])
  x_syn_list <- list()

  for (i in 1:k_class) {
    counter <- 0
    x_main <- x_classes[[i]]

    x_syn_list[[i]] <- matrix(data = NA, nrow = 0, ncol = p)
    while (TRUE) {
      if (counter == n_needed[i]) {
        break
      }
      counter <- counter + 1

      i_sample <- sample(1:n_classes[i], size = 1)
      x_main_selected <- x_main[i_sample,,drop = FALSE]

      x_syn_list[[i]] <- rbind(x_syn_list[[i]], x_main_selected)
    }
  }

  x_syn <- do.call(rbind, x_syn_list)
  y_syn <- factor(unlist(sapply(1:k_class, function(m) rep(class_names[m], n_needed[m]))), levels = class_names, labels = class_names)

  x_new <- rbind(x, x_syn)
  y_new <- c(y, y_syn)

  colnames(x_new) <- var_names
  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_syn,
    y_syn = y_syn
  ))
}

