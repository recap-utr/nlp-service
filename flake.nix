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
          entrypoint = pkgs.writeShellScriptBin "entrypoint" ''
            export LD_PRELOAD=$(${pkgs.busybox}/bin/find /lib/x86_64-linux-gnu -name "libcuda.so.*" -type f 2>/dev/null)
            exec ${lib.getExe config.packages.default} "$@"
          '';
          betterproto = pkgs.python3.pkgs.callPackage ./betterproto.nix { };
          betterproto-compiler = betterproto.overridePythonAttrs (old: {
            dependencies = old.dependencies ++ old.passthru.optional-dependencies.compiler;
          });
          pythonSet = pkgs.callPackage ./default.nix {
            inherit (inputs) uv2nix pyproject-nix pyproject-build-systems;
          };
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
          apps.default.program = pkgs.writeShellScriptBin "nlp-service" ''
            export LD_PRELOAD=/run/opengl-driver/lib/libcuda.so.1
            exec ${lib.getExe config.packages.default} "$@"
          '';
          checks = {
            inherit (config.packages) betterproto nlp-service;
            docker = config.packages.docker.passthru.stream;
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
            default = config.packages.nlp-service;
            nlp-service = pythonSet.mkApp "default";
            docker = pkgs.dockerTools.buildLayeredImage {
              name = "nlp-service";
              tag = "latest";
              created = "now";
              config.Entrypoint = [
                (lib.getExe entrypoint)
                "--host"
                "0.0.0.0"
              ];
            };
            release-env = pkgs.buildEnv {
              name = "release-env";
              paths = with pkgs; [
                uv
                python3
              ];
            };
            betterproto = pkgs.python3.pkgs.toPythonApplication betterproto-compiler;
            buf-generate = pkgs.writeShellApplication {
              name = "buf-generate";
              runtimeInputs = [ config.packages.betterproto ];
              text = ''
                TMPDIR="$(mktemp -d)"
                ${lib.getExe pkgs.buf} generate -o "$TMPDIR"
                {
                  echo "# type: ignore"
                  cat "$TMPDIR/gen/arg_services/nlp/v1/__init__.py"
                } > ./src/nlp_service/nlp_pb.py
                rm -rf "$TMPDIR"
              '';
            };
          };
          legacyPackages.docker-manifest = flocken.legacyPackages.${system}.mkDockerManifest {
            github = {
              enable = true;
              token = "$GH_TOKEN";
            };
            version = builtins.getEnv "VERSION";
            images = with self.packages; [ x86_64-linux.docker ];
          };
          devShells.default = pkgs.mkShell {
            packages = [
              pkgs.uv
              config.packages.buf-generate
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

              # if .env exists, export its variables
              if [ -f .env ]; then
                set -a
                source .env
                set +a
              fi
            '';
          };
        };
    };
}
