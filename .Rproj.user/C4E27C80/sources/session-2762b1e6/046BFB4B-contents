#' @title  Random Oversampling (ROS)
#'
#' @description Resampling with ROS.
#'
#' @param x feature matrix.
#' @param y a factor class variable with two classes.
#'
#' @details
#' Random Oversampling (ROS) is a method of copying and pasting of positive
#' samples until balance is achieved.
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
#'
#' @rdname ROS
#' @export


ROS <- function(x, y) {

  x <- as.matrix(x)
  n <- length(y)
  p <- ncol(x)

  class_names <- as.character(unique(y))
  class_pos <- names(which.min(table(y)))
  class_neg <- class_names[class_names != class_pos]

  x_pos <- x[y == class_pos,]
  x_neg <- x[y == class_neg,]

  n_pos <- nrow(x_pos)
  n_neg <- nrow(x_neg)

  imb_ratio <- n_neg/n_pos
  n_new <- (n_neg - n_pos)

  i_new <- rep(1:n_pos, ceiling(imb_ratio) - 1)

  # exact balance
  n_diff <- (n_new - length(i_new))

  i_diff <- sample(1:n_pos, n_diff)
  n_diff[i_diff] <- n_diff[i_diff] + n_diff/abs(n_diff)

  x_new <- x_pos[i_new,]

  x_new <- rbind(
    x_new,
    x_pos,
    x_neg
  )
  y_new <- c(
    rep(class_pos, n_new + n_pos),
    rep(class_neg, n_neg)
  )
  y_new <- factor(y_new, levels = levels(y), labels = levels(y))

  return(list(
    x_new = x_new,
    y_new = y_new
  ))
}

