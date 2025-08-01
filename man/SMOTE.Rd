% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SMOTE.R
\name{SMOTE}
\alias{SMOTE}
\title{Synthetic Minority Oversampling Technique (SMOTE)}
\usage{
SMOTE(x, y, k = 5, ovRate = 1)
}
\arguments{
\item{x}{feature matrix.}

\item{y}{a factor class variable with two classes.}

\item{k}{number of neighbors. Default is 5.}

\item{ovRate}{Oversampling rate multiplied by the difference between maximum
and other of class sizes. Default is 1 meaning full balance.}
}
\value{
a list with resampled dataset.
 \item{x_new}{Resampled feature matrix.}
 \item{y_new}{Resampled target variable.}
 \item{x_syn}{Generated synthetic feature data.}
 \item{y_syn}{Generated synthetic label data.}
}
\description{
Resampling with SMOTE.
}
\details{
SMOTE (Chawla et al., 2002) is an oversampling method which creates links
between positive samples and nearest neighbors and generates synthetic
samples along that link.

It is well known that SMOTE is sensitive to noisy data. It may create more
noise.

Can work with classes more than 2.

Note: Much faster than \code{smotefamily::SMOTE()}.
}
\examples{

set.seed(1)
x <- rbind(matrix(rnorm(2000, 3, 1), ncol = 2, nrow = 1000),
           matrix(rnorm(100, 5, 1), ncol = 2, nrow = 50))
y <- as.factor(c(rep("negative", 1000), rep("positive", 50)))

plot(x, col = y)

# resampling
m <- SMOTE(x = x, y = y, k = 7)

plot(m$x_new, col = m$y_new)

}
\references{
Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE:
synthetic minority over-sampling technique. Journal of artificial
intelligence research, 16, 321-357.
}
\author{
Fatih Saglam, saglamf89@gmail.com
}
