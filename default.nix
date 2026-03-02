{
  lib,
  callPackage,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  python312,
  tbb_2022,
}:
let
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
  projectOverlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };
  getCudaPkgs = attrs: lib.filter (name: lib.hasPrefix "nvidia-" name) (lib.attrNames attrs);
  cudaOverlay =
    final: prev:
    lib.genAttrs (getCudaPkgs prev) (
      name:
      prev.${name}.overrideAttrs (old: {
        autoPatchelfIgnoreMissingDeps = true;
      })
    );
  buildSystemOverlay =
    final: prev:
    lib.mapAttrs
      (
        name: value:
        prev.${name}.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ (final.resolveBuildSystem value);
        })
      )
      {
        llvmlite = {
          setuptools = [ ];
        };
      };
  packageOverlay =
    final: prev:
    lib.mapAttrs (name: value: prev.${name}.overrideAttrs value) {
      torch = old: {
        autoPatchelfIgnoreMissingDeps = true;
      };
      numba = old: {
        buildInputs = (old.buildInputs or [ ]) ++ [ tbb_2022 ];
      };
      nlp-service = old: {
        meta = (old.meta or { }) // {
          mainProgram = "nlp-service";
          maintainers = with lib.maintainers; [ mirkolenz ];
          license = lib.licenses.mit;
          homepage = "https://github.com/recap-utr/nlp-service";
          description = "Microservice for NLP tasks using gRPC";
          platforms = with lib.platforms; darwin ++ linux;
        };
      };
    };
  baseSet = callPackage pyproject-nix.build.packages {
    python = python312;
  };
  pythonSet = baseSet.overrideScope (
    lib.composeManyExtensions [
      pyproject-build-systems.overlays.default
      projectOverlay
      buildSystemOverlay
      cudaOverlay
      packageOverlay
    ]
  );
  mkVenv =
    name: deps:
    (pythonSet.mkVirtualEnv name deps).overrideAttrs (_: {
      venvIgnoreCollisions = [ "${python312.sitePackages}/griffe/*" ];
    });
  inherit (callPackage pyproject-nix.build.util { }) mkApplication;
in
mkApplication {
  venv = mkVenv "nlp-service-env" workspace.deps.optionals;
  package = pythonSet.nlp-service;
}
