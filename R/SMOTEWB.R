#' @title  SMOTE with boosting (SMOTEWB)
#'
#' @description Resampling with SMOTE with boosting.
#'
#' @param x feature matrix.
#' @param y a factor class variable with two classes.
#' @param n_weak_classifier number of weak classifiers for boosting.
#' @param class_weights numeric vector of length two. First number is for
#' positive class, and second is for negative. Higher the relative weight,
#' lesser noises for that class. By default,  \eqn{2\times n_{neg}/n} for
#' positive and \eqn{2\times n_{pos}/n} for negative class.
#' @param k_max to increase maximum number of neighbors. Default is
#' \code{ceiling(n_neg/n_pos)}.
#' @param ... additional inputs for ada::ada().
#'
#' @details
#' SMOTEWB (Saglam & Cengiz, 2022) is a SMOTE-based oversampling method which
#' can handle noisy data and adaptively decides the appropriate number of neighbors
#' to link during resampling with SMOTE.
#'
#' Trained model based on this method gives significantly better Matthew
#' Correlation Coefficient scores compared to others.
#'
#' @return a list with resampled dataset.
#'  \item{x_new}{Resampled feature matrix.}
#'  \item{y_new}{Resampled target variable.}
#'  \item{x_syn}{Generated synthetic data.}
#'  \item{w}{Boosting weights for original dataset.}
#'  \item{k}{Number of nearest neighbors for positive class samples.}
#'  \item{C}{Number of synthetic samples for each positive class samples.}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @importFrom  FNN knnx.index
#' @importFrom  stats runif
#' @importFrom  stats sd
#'
#' @references
#' Sağlam, F., & Cengiz, M. A. (2022). A novel SMOTE-based resampling technique
#' trough noise detection and the boosting procedure. Expert Systems with
#' Applications, 200, 117023.
#'
#' Can work with 2 classes only yet.
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
#' m <- SMOTEWB(x = x, y = y, n_weak_classifier = 150)
#'
#' plot(m$x_new, col = m$y_new)
#'
#'
#' @rdname SMOTEWB
#' @export

SMOTEWB <- function(
    x,
    y,
    n_weak_classifier = 100,
    class_weights = NULL,
    k_max = NULL,
    ...) {

  if (!is.data.frame(x) & !is.matrix(x)) {
    stop("x must be a matrix or dataframe")
  }

  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }

  if (!is.factor(y)) {
    stop("y must be a factor")
  }

  # var_names <- colnames(x)
  # x <- as.matrix(x)
  # p <- ncol(x)
  # n <- nrow(x)
  #
  # class_names <- levels(y)
  # n_classes <- sapply(class_names, function(m) sum(y == m))
  # k_class <- length(class_names)
  # n_classes_max <- max(n_classes)
  # n_needed <- n_classes_max - n_classes
  # x_classes <- lapply(class_names, function(m) x[y == m,, drop = FALSE])
  #
  # w <- boosted_weights(x = x, y = y, n_iter = n_weak_classifier)
  # w_classes <- lapply(class_names, function(m) w[y == class_names])
  #
  # if (is.null(class_weights)) {
  #   wclass <- n/n_classes
  # } else {
  #   wclass <- class_weights
  # }
  #
  # treshs <- (1/n)*w_class
  # scl <- sum(treshs*n_classes)
  # treshs <- treshs(scl)
  #
  # nl <- sapply(1:k_class, function(m) {
  #   ifelse(w_classes[[m]] > treshs[m], "noise", "notnoise")
  # })
  #
  # nl_classes <- lapply(class_names, function(m) nl[y == class_names])
  #
  # n_noise_classes <- lapply(nl_classes, function(m) sum(m == "noise"))
  # n_notnoise_classes <- lapply(nl_classes, function(m) sum(m == "notnoise"))
  #
  # x_classes_noise <- lapply(1:k_class, function(m) {
  #   x_classes[[m]][nl_classes[[m]] == "noise",,drop = FALSE]
  # })
  # x_classes_notnoise <- lapply(1:k_class, function(m) {
  #   x_classes[[m]][nl_classes[[m]] == "notnoise",,drop = FALSE]
  # })
  #
  #
  # x_syn_list <- list()
  #
  # for (i in 1:k_class) {
  #   counter <- 0
  #   x_main <- x_classes[[i]]
  #
  #   NN_main2main <- FNN::get.knnx(data = x_classes[[i]], query = x_classes[[i]], k = k + 1)$nn.index[,-1]
  #
  #   x_syn_list[[i]] <- matrix(data = NA, nrow = 0, ncol = p)
  #
  #   while (TRUE) {
  #     if (counter == n_needed[i]) {
  #       break
  #     }
  #     counter <- counter + 1
  #
  #     i_sample <- sample(1:n_classes[i], size = 1)
  #     x_main_selected <- x_main[i_sample,,drop = FALSE]
  #     x_target <- x_main[sample(NN_main2main[i_sample,], size = 1),,drop = FALSE]
  #     r <- runif(1)
  #
  #     x_syn_list[[i]] <- rbind(x_syn_list[[i]], x_main_selected + r*(x_target - x_main_selected))
  #   }
  # }
  #


  var_names <- colnames(x)
  x <- as.matrix(x)
  n <- length(y)
  p <- ncol(x)

  class_names <- as.character(unique(y))
  class_pos <- names(which.min(table(y)))
  class_neg <- class_names[class_names != class_pos]

  x_pos <- x[y == class_pos,,drop = FALSE]
  x_neg <- x[y == class_neg,,drop = FALSE]

  n_pos <- nrow(x_pos)
  n_neg <- nrow(x_neg)

  imb_ratio <- n_neg/n_pos

  w <- boosted_weights(x = x, y = y, n_iter = n_weak_classifier)

  w_pos <- w[y == class_pos]
  w_neg <- w[y == class_neg]

  if (is.null(class_weights)) {
    wclass_pos <- n/n_pos*0.5
    wclass_neg <- n/n_neg*0.5
  } else {
    wclass_pos <- class_weights[1]
    wclass_neg <- class_weights[2]
  }

  T_pos <- (1/n)*wclass_pos
  T_neg <- (1/n)*wclass_neg

  scl <- T_pos*n_pos + T_neg*n_neg

  T_pos <- T_pos/scl
  T_neg <- T_neg/scl

  nl_neg <- ifelse(w_neg > T_neg, "noise", "notnoise")
  nl_pos <- ifelse(w_pos > T_pos, "noise", "notnoise")

  n_neg_noise <- sum(nl_neg == "noise")
  n_pos_noise <- sum(nl_pos == "noise")
  n_neg_notnoise <- sum(nl_neg == "notnoise")
  n_pos_notnoise <- sum(nl_pos == "notnoise")

  x_neg_noise <- x_neg[nl_neg == "noise",,drop = FALSE]
  x_pos_noise <- x_pos[nl_pos == "noise",,drop = FALSE]
  x_neg_notnoise <- x_neg[nl_neg == "notnoise",,drop = FALSE]
  x_pos_notnoise <- x_pos[nl_pos == "notnoise",,drop = FALSE]

  if (is.null(k_max)) {
    k_max <- ceiling(imb_ratio)
  }
  x_notnoise <- rbind(x_pos_notnoise, x_neg_notnoise)
  y_notnoise <- c(rep(class_pos, n_pos_notnoise),
                  rep(class_neg, n_neg_notnoise))

  k_max <- min(k_max, n_pos - 2)
  NN <- FNN::knnx.index(data = x_notnoise, query = x_pos, k = k_max + 1)
  NN_temp <- matrix(data = NA, nrow = n_pos, ncol = k_max)
  NN_temp[nl_pos == "noise", ] <- NN[nl_pos == "noise", -(k_max + 1)]
  NN_temp[nl_pos == "notnoise", ] <- NN[nl_pos == "notnoise", -1]
  NN <- NN_temp

  k <- c()
  fl <- c()

  for (i in 1:n_pos) {
    cls <- y_notnoise[NN[i,]]

    if (all(cls == class_pos)) {
      k[i] <- k_max
    } else {
      k[i] <- which(cls == class_neg)[1] - 1
    }

    if (k[i] == 0 & nl_pos[i] == "noise") {
      fl[i] <- "bad"
    }
    if (k[i] == 0 & nl_pos[i] == "notnoise") {
      fl[i] <- "lonely"
    }
    if (k[i] > 0) {
      fl[i] <- "good"
    }
  }

  n_syn <- (n_neg - n_pos)
  C <- numeric(n_pos)
  n_good_and_lonely <- sum((fl == "good") + (fl == "lonely"))
  for (i in 1:n_pos) {
    if (fl[i] == "good" | fl[i] == "lonely") {
      C[i] <- ceiling(n_syn/n_good_and_lonely)
    }
  }
  n_diff <- (n_syn - sum(C))

  ii <- sample(which(fl == "good" | fl == "lonely"), size = abs(n_diff))
  C[ii] <- C[ii] + n_diff/abs(n_diff)

  x_syn <- matrix(nrow = 0, ncol = p)
  for (i in 1:n_pos) {
    if (fl[i] == "lonely") {
      i_step <- rep(i, C[i])
      x_syn_step <- x_pos[i_step,]
      x_syn <- rbind(x_syn, x_syn_step)
    }
    if (fl[i] == "good") {
      if (C[i] == 0) {
        next
      }
      NN_i <- NN[i,1:k[i]]
      i_k <- sample(1:k[i], C[i], replace = TRUE)
      lambda <- runif(C[i])
      kk <- x_notnoise[NN_i,,drop = FALSE]
      kk <- kk[i_k,]
      x_pos_i_temp <- x_pos[rep(i, C[i]),,drop = FALSE]
      x_syn_step <- x_pos_i_temp + (kk - x_pos_i_temp)*lambda
      x_syn <- rbind(x_syn, x_syn_step)
    }
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
    w = w,
    k = k,
    C = C
  ))
}
