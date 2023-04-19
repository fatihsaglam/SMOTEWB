#' @title  Synthetic Minority Oversampling Technique (SMOTE)
#'
#' @description Resampling with SMOTE.
#'
#' @param x feature matrix.
#' @param y a factor class variable with two classes.
#' @param k number of neighbors. Default is 5.
#'
#' @details
#' SMOTE (Chawla et al., 2002) is an oversampling method which creates links
#' between positive samples and nearest neighbors and generates synthetic
#' samples along that link.
#'
#' It is well known that SMOTE is sensitive to noisy data. It may create more
#' noise.
#'
#' Note: Much faster than \code{smotefamily::SMOTE()}.
#'
#' @return a list with resampled dataset.
#'  \item{x_new}{Resampled feature matrix.}
#'  \item{y_new}{Resampled target variable.}
#'  \item{x_syn}{Generated synthetic data.}
#'  \item{C}{Number of synthetic samples for each positive class samples.}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @importFrom  FNN knnx.index
#' @importFrom  stats runif
#' @importFrom  stats sd
#'
#' @references
#' Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE:
#' synthetic minority over-sampling technique. Journal of artificial
#' intelligence research, 16, 321-357.
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
#' m <- SMOTE(x = x, y = y, k = 7)
#'
#' plot(m$x_new, col = m$y_new)
#'
#' @rdname SMOTE
#' @export

SMOTE <- function(x, y, k = 5) {

  if (!is.data.frame(x) & !is.matrix(x)) {
    stop("x must be a matrix or dataframe")
  }

  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }

  if (!is.factor(y)) {
    stop("y must be a factor")
  }

  if (!is.numeric(k)) {
    stop("k must be numeric")
  }

  if (k < 1) {
    stop("k must be positive")
  }

  var_names <- colnames(x)
  x <- as.matrix(x)
  p <- ncol(x)

  class_names <- as.character(unique(y))
  class_pos <- names(which.min(table(y)))
  class_neg <- class_names[class_names != class_pos]

  x_pos <- x[y == class_pos,,drop = FALSE]
  x_neg <- x[y == class_neg,,drop = FALSE]

  n_pos <- nrow(x_pos)
  n_neg <- nrow(x_neg)

  imb_ratio <- n_neg/n_pos
  k <- min(k, n_pos - 1)
  NN <- FNN::knnx.index(data = x_pos, query = x_pos, k = k + 1)[, -1]

  n_syn <- (n_neg - n_pos)
  C <- rep(ceiling(imb_ratio) - 1, n_pos)

  n_diff <- (n_syn - sum(C))

  ii <- sample(1:n_pos, size = abs(n_diff))
  C[ii] <- C[ii] + n_diff/abs(n_diff)

  x_syn <- matrix(nrow = 0, ncol = p)
  for (i in 1:n_pos) {
    if (C[i] == 0) {
      next
    }
    NN_i <- NN[i,]
    i_k <- sample(1:k, C[i], replace = TRUE)
    lambda <- runif(C[i])
    kk <- x_pos[NN_i,,drop = FALSE]
    kk <- kk[i_k,]
    x_pos_i_temp <- x_pos[rep(i, C[i]),,drop = FALSE]
    x_syn_step <- x_pos_i_temp + (kk - x_pos_i_temp)*lambda
    x_syn <- rbind(x_syn, x_syn_step)
  }

  x_new <- rbind(
    x_syn,
    x_pos,
    x_neg
  )
  y_new <- c(
    rep(class_pos, n_syn + n_pos),
    rep(class_neg, n_neg)
  )
  y_new <- factor(y_new, levels = levels(y), labels = levels(y))
  colnames(x_new) <- var_names

  return(list(
    x_new = x_new,
    y_new = y_new,
    x_syn = x_new[1:n_syn,, drop = FALSE],
    C = C
  ))

}
