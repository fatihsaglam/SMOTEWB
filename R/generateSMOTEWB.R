#' @title Data generation for SMOTE with Boosting (SMOTEWB)
#'
#' @description Data generation for SMOTE with Boosting (SMOTEWB)
#'
#' @param x_pos positive class feature matrix.
#' @param x_neg negative class feature matrix.
#' @param n_syn number of synthetic samples to generate.
#' @param k_max maximum number of nearest neighbors.
#' @param n_weak_classifier number of weak classifiers.
#' @param class_pos label of the positive class.
#' @param class_names class labels.
#' @param class_weights class weights.
#'
#' @details
#' Data generation for SMOTE with Boosting.
#'
#' @return a vector of case weights.
#'  \item{x_syn}{last weight vector of boosting process}
#'  \item{y_syn}{Generated synthetic data labels.}
#'  \item{w}{Boosting weights for original dataset.}
#'  \item{k}{Number of nearest neighbors for positive class samples.}
#'  \item{C}{Number of synthetic samples for each positive class samples.}
#'  \item{fl}{"good", "bad" and "lonely" sample labels}
#'
#' @author Fatih Saglam, saglamf89@gmail.com
#'
#' @noRd

generateSMOTEWB <- function(x_pos,
                            x_neg,
                            n_syn,
                            k_max,
                            n_weak_classifier,
                            class_pos,
                            class_names,
                            class_weights) {
  if (n_syn == 0) {
    return(list(
      x_syn = matrix(data = NA, nrow = 0, ncol = ncol(x_pos)),
      y_syn = factor(c(), levels = class_names),
      C = list(),
      w = list(),
      k = list(),
      fl = list()
    ))
  }

  class_neg <- paste0(class_names[class_names != class_pos], collapse = "-")

  n_pos <- nrow(x_pos)
  n_neg <- nrow(x_neg)
  n <- n_pos + n_neg
  p <- ncol(x_pos)

  x <- rbind(x_pos, x_neg)
  y <- factor(c(rep(class_pos, n_pos), rep(class_neg, n_neg)), levels = c(class_pos, class_neg))

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

  k_max <- min(k_max, nrow(x_notnoise) - 1)
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

  return(list(
    x_syn = x_syn,
    y_syn = factor(rep(class_pos, nrow(x_syn)), class_names),
    C = C,
    w = w,
    k = k,
    fl = fl
  ))
}
