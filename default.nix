{
  lib,
  callPackage,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  python3,
}:
let
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
  projectOverlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };
  getCudaPkgs = attrs: lib.filter (name: lib.hasPrefix "nvidia-" name) (lib.attrNames attrs);
  cudaOverlay =
    final: prev:
    lib.genAttrs ([ "torch" ] ++ (getCudaPkgs prev)) (
      name:
      prev.${name}.overrideAttrs (old: {
        autoPatchelfIgnoreMissingDeps = true;
      })
    );
  baseSet = callPackage pyproject-nix.build.packages {
    python = python3;
  };
  pythonSet = baseSet.overrideScope (
    lib.composeManyExtensions [
      pyproject-build-systems.overlays.default
      projectOverlay
      cudaOverlay
    ]
  );
  addMeta =
    drv:
    drv.overrideAttrs (old: {
      meta = (old.meta or { }) // {
        mainProgram = "nlp-service";
        maintainers = with lib.maintainers; [ mirkolenz ];
        license = lib.licenses.mit;
        homepage = "https://github.com/recap-utr/nlp-service";
        description = "Microservice for NLP tasks using gRPC";
        platforms = with lib.platforms; darwin ++ linux;
      };
    });
in
pythonSet
// {
  mkApp = depsName: addMeta (pythonSet.mkVirtualEnv "nlp-service-env" workspace.deps.${depsName});
}
