# dyn-cc
## Dynamic Coreset Clustering Development

The goal is to eventually implement [this paper](https://openreview.net/forum?id=FQ2dMjf88y&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)) on fully dynamic coreset spectral clustering in a way that is compatible (or inside of) [Raphtory](https://github.com/pometry/raphtory), while making the implementation agnostic over the choice of static clustering algorithm, numerically robust, and capable of handling batch updates. If wildly succesful, I'll have a go at a distributed version.

In this repo, I'm building the backbone for doing this that I'll be able to integrate with raphtory once I've worked out the best way of interfacing with it.
