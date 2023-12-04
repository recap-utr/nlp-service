# https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/python-modules/betterproto/default.nix
{
  fetchFromGitHub,
  lib,
  python,
  buildPythonPackage,
  poetry-core,
  grpclib,
  python-dateutil,
  typing-extensions,
  black,
  jinja2,
  isort,
  pytestCheckHook,
  pytest-asyncio,
  pytest-cov,
  pytest-mock,
  pydantic,
  protobuf,
  cachelib,
  tomlkit,
  grpcio-tools,
}:
buildPythonPackage rec {
  pname = "betterproto";
  version = "master";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "danielgtaylor";
    repo = "python-betterproto";
    rev = "bd7de203e16e949666b2844b3dec1eb7c4ed523c";
    hash = "sha256-ppVS8dfVSXBm7KGv1/um6ePK4pBln+RrizR9EXz40qo=";
  };

  nativeBuildInputs = [poetry-core];

  propagatedBuildInputs = [
    grpclib
    python-dateutil
    typing-extensions
  ];

  passthru.optional-dependencies.compiler = [
    black
    jinja2
    isort
  ];

  nativeCheckInputs =
    [
      pytestCheckHook
      pytest-asyncio
      pytest-cov
      pytest-mock
      pydantic
      protobuf
      cachelib
      tomlkit
      grpcio-tools
    ]
    ++ passthru.optional-dependencies.compiler;

  # The tests require the generation of code before execution. This requires
  # the protoc-gen-python_betterproto script from the package to be on PATH.
  preCheck = ''
    export PATH=$PATH:$out/bin
    ${python.interpreter} -m tests.generate
  '';

  pythonImportsCheck = ["betterproto"];

  meta = {
    description = "Clean, modern, Python 3.6+ code generator & library for Protobuf 3 and async gRPC";
    longDescription = ''
      This project aims to provide an improved experience when using Protobuf /
      gRPC in a modern Python environment by making use of modern language
      features and generating readable, understandable, idiomatic Python code.
    '';
    homepage = "https://github.com/danielgtaylor/python-betterproto";
    license = lib.licenses.mit;
  };
}
