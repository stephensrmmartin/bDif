% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/bDif.R
\name{compareMethods}
\alias{compareMethods}
\title{Compare DIF detection with methods from difR package}
\usage{
compareMethods(object, chains = object@chain.max, groups = clusters(object,
  modal = TRUE, chains), focal.name = 1, method = c("TID", "BD", "Raju",
  "MH", "Logistic", "Std", "Lord"), ...)
}
\arguments{
\item{object}{A bDif object returned from \code{\link{bDifOut}}.}

\item{chains}{A numeric vector indicating the mcmc chain(s).}

\item{groups}{Vector containing group memberships. See \link[difR]{dichoDif}.}

\item{focal.name}{Name of reference group. See \link[difR]{dichoDif}.}

\item{method}{Character vector. Vector of DIF detection methods from the difR package. See \link[difR]{dichoDif}.}

\item{...}{Not currently used.}
}
\value{
A dichoDif object. See \link[difR]{dichoDif}
}
\description{
Compare DIF detection with methods from difR package
}

