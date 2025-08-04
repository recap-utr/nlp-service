# Changelog

## [2.0.3](https://github.com/recap-utr/nlp-service/compare/v2.0.2...v2.0.3) (2025-08-04)

### Bug Fixes

* add autoload member and load method ([83e4aa1](https://github.com/recap-utr/nlp-service/commit/83e4aa1fe9ed8137745182ec1f8c7f6ef65a9028))
* add control over embed provider cache behavior ([832c04a](https://github.com/recap-utr/nlp-service/commit/832c04ab555dee29fab1cf441307681e089f9cbb))
* use sqlite for vector cache ([71a3ed9](https://github.com/recap-utr/nlp-service/commit/71a3ed9e1bbe2b29f0c184171b5d2ef6eb777d68))

## [2.0.2](https://github.com/recap-utr/nlp-service/compare/v2.0.1...v2.0.2) (2025-05-20)

### Bug Fixes

* **build:** correctly wrap package to expose cuda libraries ([0bc9bf7](https://github.com/recap-utr/nlp-service/commit/0bc9bf78a8405a66ae81980ad7848facf543254e))

## [2.0.1](https://github.com/recap-utr/nlp-service/compare/v2.0.0...v2.0.1) (2025-03-03)

### Bug Fixes

* add tensorflow-hub embedder ([f325e19](https://github.com/recap-utr/nlp-service/commit/f325e190758967028748eef8db0e2bc293c75311))

## [2.0.0](https://github.com/recap-utr/nlp-service/compare/v1.4.10...v2.0.0) (2025-02-06)

### ⚠ BREAKING CHANGES

* The NLP service is now based on [CBRkit](https://github.com/wi2trier/cbrkit).
* The NLP service can be imported as a Python library to convert Protobuf-based NLP configurations into CBRkit similarity functions.

### Features

* add autodump ([ee0fcb2](https://github.com/recap-utr/nlp-service/commit/ee0fcb267385514dbba4e667f9cbe569c942fafb))
* migrate to central class-based interface ([734b7dc](https://github.com/recap-utr/nlp-service/commit/734b7dc7ef9b41d0fb5e1913014ad860e38c2f06))
* remove global state and make the nlp config a parameter ([9d54af7](https://github.com/recap-utr/nlp-service/commit/9d54af7a4c367bced26f97153cc51dd912986846))
* rewrite the service completely ([07a53c6](https://github.com/recap-utr/nlp-service/commit/07a53c6de61cfde496634afaa50c31564504fd06))
* rewrite the service using cbrkit as the backend ([ac78b54](https://github.com/recap-utr/nlp-service/commit/ac78b54991fd8c4833743e895f2c15b51cfe6385))
* switch back to google's grpc/protobuf due to compatibility issues ([87d380d](https://github.com/recap-utr/nlp-service/commit/87d380d0f58ba0bc482d3156b6f24900fe9333bb))

### Bug Fixes

* remove lazy from cache ([5d42fa8](https://github.com/recap-utr/nlp-service/commit/5d42fa84fbbce9751c8ba658a5dd0bafa4b515bb))
* remove specialized embedders, always build on similarity ([13d8304](https://github.com/recap-utr/nlp-service/commit/13d83043d1f367333470aca5f402cf919ee7db9b))
* **server:** set up logging ([ad51b9d](https://github.com/recap-utr/nlp-service/commit/ad51b9df697ea7393a69983d5903ede2fb36367c))
* use correct data types for serializable embedding model ([a97fe3a](https://github.com/recap-utr/nlp-service/commit/a97fe3a538994f6a6b702fb5a12d1345014ce2dd))

### Miscellaneous Chores

* update changelog ([d81c34c](https://github.com/recap-utr/nlp-service/commit/d81c34c427b67d533e1b98c3da7f9814a693625d))

## [2.0.0-beta.2](https://github.com/recap-utr/nlp-service/compare/v2.0.0-beta.1...v2.0.0-beta.2) (2024-05-28)


### Bug Fixes

* use correct data types for serializable embedding model ([a97fe3a](https://github.com/recap-utr/nlp-service/commit/a97fe3a538994f6a6b702fb5a12d1345014ce2dd))

## [2.0.0-beta.1](https://github.com/recap-utr/nlp-service/compare/v1.4.10...v2.0.0-beta.1) (2023-12-04)


### ⚠ BREAKING CHANGES

* Instead of using protobuf definitions from arg-services, the package now contains code generated via betterproto. We also added a library that can be used in other Python projects without needing to start a server. Lastly, a proper CLI has been integrated that (besides starting the server) allows to perform semantic retrieval in the local file system.

### Features

* rewrite the service completely ([07a53c6](https://github.com/recap-utr/nlp-service/commit/07a53c6de61cfde496634afaa50c31564504fd06))

## [1.4.10](https://github.com/recap-utr/nlp-service/compare/v1.4.9...v1.4.10) (2023-11-07)


### Bug Fixes

* update openai library to v1 ([8493556](https://github.com/recap-utr/nlp-service/commit/84935568da53aae271464ae7fbfc68432fe4e700))

## [1.4.9](https://github.com/recap-utr/nlp-service/compare/v1.4.8...v1.4.9) (2023-07-04)


### Bug Fixes

* set ld_preload in docker entrypoint ([1285bf5](https://github.com/recap-utr/nlp-service/commit/1285bf5edca2a0190d89d413bd89cd14e0757c07))

## [1.4.8](https://github.com/recap-utr/nlp-service/compare/v1.4.7...v1.4.8) (2023-07-04)


### Bug Fixes

* add ld_library_path to docker image ([9d75c4b](https://github.com/recap-utr/nlp-service/commit/9d75c4bc3290685f3619255563ff660401d71f70))

## [1.4.7](https://github.com/recap-utr/nlp-service/compare/v1.4.6...v1.4.7) (2023-07-03)


### Bug Fixes

* trigger rebuild ([93f1f93](https://github.com/recap-utr/nlp-service/commit/93f1f9389d370a3880a6e8b5bf635a5b6a65e066))

## [1.4.6](https://github.com/recap-utr/nlp-service/compare/v1.4.5...v1.4.6) (2023-07-02)


### Bug Fixes

* **ci:** add correct permissions ([ed5724f](https://github.com/recap-utr/nlp-service/commit/ed5724fe6565c75fc66134e55452391319b659a8))

## [1.4.5](https://github.com/recap-utr/nlp-service/compare/v1.4.4...v1.4.5) (2023-07-02)


### Bug Fixes

* **ci:** automatically build docker image ([9bbdc77](https://github.com/recap-utr/nlp-service/commit/9bbdc7788d49dc85cbf32a9a4ba14c2cf56674c4))

## [1.4.4](https://github.com/recap-utr/nlp-service/compare/v1.4.3...v1.4.4) (2023-06-29)


### Bug Fixes

* add ld_preload to nix flake ([45070d7](https://github.com/recap-utr/nlp-service/commit/45070d7a8adae7b312354f35272e3aa84c5c2ad3))
* download spacy models to custom cache ([43013d8](https://github.com/recap-utr/nlp-service/commit/43013d8b7a1fdd8b41e1d72d889c8055a40bb646))
* remove tensorflow due to cuda issues ([08dfae0](https://github.com/recap-utr/nlp-service/commit/08dfae09f5c86528a8a08015a2c0a119bc340e1e))
* replace dockerfiles with nix-based build ([be19aff](https://github.com/recap-utr/nlp-service/commit/be19affa32a2bc200fb353556c049345649c31e9))

## [1.4.3](https://github.com/recap-utr/nlp-service/compare/v1.4.2...v1.4.3) (2023-06-07)


### Bug Fixes

* update deps to remove yanked grpcio release ([4b1f277](https://github.com/recap-utr/nlp-service/commit/4b1f2773953f812925de57b181cfb58f7e25d7f7))

## [1.4.2](https://github.com/recap-utr/nlp-service/compare/v1.4.1...v1.4.2) (2023-06-07)


### Bug Fixes

* allow specifying threads ([f67af0c](https://github.com/recap-utr/nlp-service/commit/f67af0c55e09f261bafad6161a97574ab1afbd34))

## [1.4.1](https://github.com/recap-utr/nlp-service/compare/v1.4.0...v1.4.1) (2023-05-23)


### Bug Fixes

* bump deps and relax typer constraints ([90193af](https://github.com/recap-utr/nlp-service/commit/90193af6b6d5ac1a0684032190a06107ffda3816))

## [1.4.0](https://github.com/recap-utr/nlp-service/compare/v1.3.11...v1.4.0) (2023-04-10)


### Features

* add support for openai embeddings ([3779c7d](https://github.com/recap-utr/nlp-service/commit/3779c7dbebd798d10cefe5b5cff6fd6fd68eec50))


### Bug Fixes

* did not check properly for empty spacy model ([770e4d9](https://github.com/recap-utr/nlp-service/commit/770e4d91112bd6dbf19b28a488c0c07d34d046db))

## [1.3.11](https://github.com/recap-utr/nlp-service/compare/v1.3.10...v1.3.11) (2023-03-26)


### Bug Fixes

* re-enable tensorflow-hub embeddings ([188108f](https://github.com/recap-utr/nlp-service/commit/188108fdf0b862735bfc0e49dbe3e4e825c49d2c))

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
