{
  lib,
  callPackage,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  python312,
  tbb_2021,
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
  packageOverlay =
    final: prev:
    lib.mapAttrs (name: value: prev.${name}.overrideAttrs value) {
      torch = old: {
        autoPatchelfIgnoreMissingDeps = true;
      };
      numba = old: {
        buildInputs = (old.buildInputs or [ ]) ++ [ tbb_2021 ];
      };
      cbrkit = old: {
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
in
{
  inherit workspace;
  inherit (callPackage pyproject-nix.build.util { }) mkApplication;
  pythonSet = baseSet.overrideScope (
    lib.composeManyExtensions [
      pyproject-build-systems.overlays.default
      projectOverlay
      cudaOverlay
      packageOverlay
    ]
  );
}
