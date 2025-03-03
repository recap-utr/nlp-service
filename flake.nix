{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
    flocken = {
      url = "github:mirkolenz/flocken/v2";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://recap.cachix.org"
      "https://pyproject-nix.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "recap.cachix.org-1:KElwRDtaJbbQxmmS2SyxWHqs9bdJbaZHzb2iINTfQws="
      "pyproject-nix.cachix.org-1:UNzugsOlQIu2iOz0VyZNBQm2JSrL/kwxeCcFGw+jMe0="
    ];
  };
  outputs =
    inputs@{
      self,
      nixpkgs,
      flake-parts,
      systems,
      flocken,
      uv2nix,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import systems;
      imports = with inputs; [
        flake-parts.flakeModules.easyOverlay
        treefmt-nix.flakeModule
      ];
      perSystem =
        {
          pkgs,
          lib,
          system,
          config,
          ...
        }:
        let
          inherit
            (pkgs.callPackage ./default.nix {
              inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
            })
            pythonSet
            mkApplication
            workspace
            ;
        in
        {
          _module.args.pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
            overlays = lib.singleton (
              final: prev: {
                python3 = final.python312;
                uv = uv2nix.packages.${system}.uv-bin;
              }
            );
          };
          checks = {
            inherit (config.packages) nlp-service docker;
          };
          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              ruff-check.enable = true;
              ruff-format.enable = true;
              nixfmt.enable = true;
            };
          };
          packages = {
            default = config.packages.nlp-service-wrapped;
            nlp-service = mkApplication {
              venv = pythonSet.mkVirtualEnv "nlp-service-env" workspace.deps.optionals;
              package = pythonSet.nlp-service;
            };
            nlp-service-wrapped =
              pkgs.runCommand "nlp-service-wrapped"
                {
                  nativeBuildInputs = with pkgs; [ makeWrapper ];
                  inherit (config.packages.nlp-service) meta;
                }
                ''
                  mkdir -p $out/bin
                  makeWrapper \
                    ${lib.getExe config.packages.nlp-service} \
                    $out/bin/nlp-service \
                    --set LD_PRELOAD $(${pkgs.busybox}/bin/find \
                    /run/opengl-driver/lib /lib/x86_64-linux-gnu \
                    -name "libcuda.so.*" \
                    -type f \
                    2>/dev/null \
                    | head -n 1)
                '';
            docker = pkgs.dockerTools.streamLayeredImage {
              name = "nlp-service";
              tag = "latest";
              created = "now";
              config.Entrypoint = [
                (lib.getExe config.packages.default)
                "--host"
                "0.0.0.0"
              ];
              config.ExposedPorts = {
                "50100/tcp" = { };
              };
            };
            release-env = pkgs.buildEnv {
              name = "release-env";
              paths = with pkgs; [
                uv
                python3
              ];
            };
          };
          legacyPackages.docker-manifest = flocken.legacyPackages.${system}.mkDockerManifest {
            github = {
              enable = true;
              token = "$GH_TOKEN";
            };
            version = builtins.getEnv "VERSION";
            imageStreams = with self.packages; [ x86_64-linux.docker ];
          };
          devShells.default = pkgs.mkShell {
            packages = [
              pkgs.uv
              config.treefmt.build.wrapper
            ];
            nativeBuildInputs = with pkgs; [ zlib ];
            LD_LIBRARY_PATH = lib.makeLibraryPath [
              pkgs.stdenv.cc.cc
              pkgs.zlib
              "/run/opengl-driver"
            ];
            UV_PYTHON = lib.getExe pkgs.python3;
            shellHook = ''
              uv sync --all-extras --locked
            '';
          };
        };
    };
}
