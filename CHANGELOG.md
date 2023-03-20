# Changelog

## [1.3.10](https://github.com/recap-utr/nlp-service/compare/v1.3.9...v1.3.10) (2023-03-20)


### Bug Fixes

* **deps:** update dependency transformers to v4.27.2 ([#19](https://github.com/recap-utr/nlp-service/issues/19)) ([eba077e](https://github.com/recap-utr/nlp-service/commit/eba077ea787fc30411be1c214a5eae9f7a8d6d61))

## [1.3.9](https://github.com/recap-utr/nlp-service/compare/v1.3.8...v1.3.9) (2023-03-19)


### Bug Fixes

* allow pytorch v2 ([5fa6b5a](https://github.com/recap-utr/nlp-service/commit/5fa6b5a7b41516eb8e0c0c50c2c97e44953b5c3d))

## [1.3.8](https://github.com/recap-utr/nlp-service/compare/v1.3.7...v1.3.8) (2023-03-19)


### Bug Fixes

* **deps:** update dependency spacy to v3.5.1 ([#16](https://github.com/recap-utr/nlp-service/issues/16)) ([6a3f9d5](https://github.com/recap-utr/nlp-service/commit/6a3f9d534395a850877e9baec56a98bbd1318436))
* **deps:** update dependency transformers to v4.27.1 ([#17](https://github.com/recap-utr/nlp-service/issues/17)) ([fd3782b](https://github.com/recap-utr/nlp-service/commit/fd3782bb1fcd420346f233cf3f4bf03c2a9a0fc6))

## [1.3.7](https://github.com/recap-utr/nlp-service/compare/v1.3.6...v1.3.7) (2023-03-18)


### Bug Fixes

* **deps:** update dependency gensim to v4.3.1 ([#15](https://github.com/recap-utr/nlp-service/issues/15)) ([a7e7476](https://github.com/recap-utr/nlp-service/commit/a7e7476212ecb4878963a8dddc3122e92baff2e7))

## [1.3.6](https://github.com/recap-utr/nlp-service/compare/v1.3.5...v1.3.6) (2023-03-18)


### Bug Fixes

* **deps:** update dependency arg-services to v1.2.3 ([#14](https://github.com/recap-utr/nlp-service/issues/14)) ([597f498](https://github.com/recap-utr/nlp-service/commit/597f4988d2560fb35130c741f37601dec3d36612))

## [1.3.5](https://github.com/recap-utr/nlp-service/compare/v1.3.4...v1.3.5) (2023-03-06)


### Bug Fixes

* use cosine for unspecified similarity ([8619293](https://github.com/recap-utr/nlp-service/commit/861929344bae2bc865f4efa48d5edb2072c07d6b))

## [1.3.4](https://github.com/recap-utr/nlp-service/compare/v1.3.3...v1.3.4) (2023-03-06)


### Bug Fixes

* loosen numpy/scipy version constraints ([f860095](https://github.com/recap-utr/nlp-service/commit/f860095e6a11f0a4da5f2b5e48d3be8538457bbc))

## [1.3.3](https://github.com/recap-utr/nlp-service/compare/v1.3.2...v1.3.3) (2023-03-03)


### Bug Fixes

* bump deps ([a2cb819](https://github.com/recap-utr/nlp-service/commit/a2cb81929ef67d68a3f45655b17db5cc33ae5856))

## [1.3.2](https://github.com/recap-utr/nlp-service/compare/v1.3.1...v1.3.2) (2023-02-22)


### Bug Fixes

* update default port ([09b6a3c](https://github.com/recap-utr/nlp-service/commit/09b6a3c4c40b262208bea6bbbab452711ffc7ac7))

## [1.3.1](https://github.com/recap-utr/nlp-service/compare/v1.3.0...v1.3.1) (2023-02-03)


### Bug Fixes

* **docker:** add entrypoint and extras build arg ([2ad2877](https://github.com/recap-utr/nlp-service/commit/2ad287760763f90f6845f16c52254865bd81e16f))
* **server:** address should be a cli argument ([3c74b56](https://github.com/recap-utr/nlp-service/commit/3c74b5688a7f98695186df012c3140601dcd591f))
* **server:** download spacy model if not available ([defb340](https://github.com/recap-utr/nlp-service/commit/defb3405d8de14e79a412c106ff15ffa371c1a6a))

## [1.3.0](https://github.com/recap-utr/nlp-service/compare/v1.2.2...v1.3.0) (2023-02-03)


### Features

* make transformer dependencies optional ([c34f62f](https://github.com/recap-utr/nlp-service/commit/c34f62f54110a3204de94081a1a717824a73a133))

## [1.2.2](https://github.com/recap-utr/nlp-service/compare/v1.2.1...v1.2.2) (2023-01-16)


### Bug Fixes

* bump deps ([074733b](https://github.com/recap-utr/nlp-service/commit/074733b108a480dcd4bee0afc29df75c706f0ff2))
* temporarily disable docker publish workflow ([394ff40](https://github.com/recap-utr/nlp-service/commit/394ff40f85eb7976e0c265c093e4e70a1cd8790b))

## [1.2.1](https://github.com/recap-utr/nlp-service/compare/v1.2.0...v1.2.1) (2023-01-11)


### Bug Fixes

* remove server from init ([435772d](https://github.com/recap-utr/nlp-service/commit/435772d1b5e100a23f48d18a5c9212bcb49bb18d))

## [1.2.0](https://github.com/recap-utr/nlp/compare/v1.1.1...v1.2.0) (2023-01-10)


### Features

* **server:** add cache for spacy docs ([107e6b9](https://github.com/recap-utr/nlp/commit/107e6b93d14e9dd472bee80d5b673201bcc6e05a))

## [1.1.1](https://github.com/recap-utr/nlp/compare/v1.1.0...v1.1.1) (2023-01-06)


### Bug Fixes

* replace dataclasses-json with mashumaro ([bccf2ac](https://github.com/recap-utr/nlp/commit/bccf2ac3083b0704bdc7de7107054bd7c263db91))

## [1.1.0](https://github.com/recap-utr/nlp/compare/v1.0.2...v1.1.0) (2023-01-05)


### Features

* add huggingface transformers model ([e60da2b](https://github.com/recap-utr/nlp/commit/e60da2b6d289fcdaa432f92178256c8458617a74))
* add public interface in init ([bc4cf24](https://github.com/recap-utr/nlp/commit/bc4cf246c5cba4fbaedf29263c6612bebb8b9f8f))
* enforce stricter typing ([e6f49b0](https://github.com/recap-utr/nlp/commit/e6f49b09be6fdc3c34897f5dc2d899da22766dd2))

## [1.0.2](https://github.com/recap-utr/nlp/compare/v1.0.1...v1.0.2) (2022-12-20)


### Bug Fixes

* update deps ([9426a15](https://github.com/recap-utr/nlp/commit/9426a1539a07e7628d10d662d0b7dec31a30dbde))

## [1.0.1](https://github.com/recap-utr/nlp/compare/v1.0.0...v1.0.1) (2022-12-20)


### Bug Fixes

* update deps ([faa359f](https://github.com/recap-utr/nlp/commit/faa359f0a0b979c1e961a9d1483d4e6dbf3ff9c0))

## [1.0.0](https://github.com/recap-utr/nlp/compare/v0.3.5...v1.0.0) (2022-12-19)


### ⚠ BREAKING CHANGES

* bump version for proper release management

### Features

* bump version for proper release management ([b333934](https://github.com/recap-utr/nlp/commit/b33393469aeb2369122998416a412bf371d52c1e))

## [0.3.5](https://github.com/recap-utr/nlp/compare/v0.3.4...v0.3.5) (2022-12-19)


### Bug Fixes

* **deps:** update dependency arg-services to v1 ([#8](https://github.com/recap-utr/nlp/issues/8)) ([2584518](https://github.com/recap-utr/nlp/commit/2584518262c9d369b28350b6ce44fbd4ae6f80e1))

## [0.3.4](https://github.com/recap-utr/nlp/compare/v0.3.3...v0.3.4) (2022-12-13)


### Bug Fixes

* **server:** correct check for empty attributes ([36f5a08](https://github.com/recap-utr/nlp/commit/36f5a08d9e4ddce640f6ae5b011849ae35cd0571))

## [0.3.3](https://github.com/recap-utr/nlp/compare/v0.3.2...v0.3.3) (2022-12-13)


### Bug Fixes

* update dependencies to the latest versions ([8fe8138](https://github.com/recap-utr/nlp/commit/8fe813843812cce29578afc07504ab8b9bd158f1))
