{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flocken = {
      url = "github:mirkolenz/flocken/v2";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = inputs @ {
    self,
    nixpkgs,
    flake-parts,
    systems,
    poetry2nix,
    flocken,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = import systems;
      perSystem = {
        pkgs,
        lib,
        system,
        self',
        ...
      }: let
        python = pkgs.python311;
        poetry = pkgs.poetry;
        entrypoint = pkgs.writeShellScriptBin "entrypoint" ''
          export LD_PRELOAD=$(${pkgs.busybox}/bin/find /lib/x86_64-linux-gnu -name "libcuda.so.*" -type f 2>/dev/null)
          exec ${lib.getExe self'.packages.default} "$@"
        '';
        betterproto = python.pkgs.callPackage ./betterproto.nix {};
        betterprotoCompiler = betterproto.overridePythonAttrs (old: {
          propagatedBuildInputs = old.propagatedBuildInputs ++ old.passthru.optional-dependencies.compiler;
        });
      in {
        _module.args.pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
          overlays = [poetry2nix.overlays.default];
        };
        apps = {
          default = {
            type = "app";
            program = lib.getExe (pkgs.writeShellScriptBin "nlp-service" ''
              export LD_PRELOAD=/run/opengl-driver/lib/libcuda.so.1
              exec ${lib.getExe self'.packages.default} "$@"
            '');
          };
        };
        packages = {
          default = pkgs.poetry2nix.mkPoetryApplication {
            inherit python;
            projectDir = ./.;
            preferWheels = true;
          };
          docker = pkgs.dockerTools.buildImage {
            name = "nlp-service";
            tag = "latest";
            created = "now";
            config = {
              entrypoint = [(lib.getExe entrypoint)];
              cmd = ["0.0.0.0:50100"];
            };
          };
          releaseEnv = pkgs.buildEnv {
            name = "release-env";
            paths = [poetry python];
          };
          betterproto = python.pkgs.toPythonApplication betterprotoCompiler;
          bufGenerate = pkgs.writeShellApplication {
            name = "buf-generate";
            runtimeInputs = [self'.packages.betterproto];
            text = ''
              ${lib.getExe pkgs.buf} generate buf.build/recap/arg-services
              {
                echo "# type: ignore"
                cat ./gen/arg_services/nlp/v1/__init__.py
              } > ./nlp_service/nlp_pb.py
              rm -rf ./gen
            '';
          };
        };
        legacyPackages.dockerManifest = flocken.legacyPackages.${system}.mkDockerManifest {
          github = {
            enable = true;
            token = "$GH_TOKEN";
          };
          version = builtins.getEnv "VERSION";
          images = with self.packages; [x86_64-linux.docker];
        };
        devShells.default = pkgs.mkShell {
          packages = [poetry python self'.packages.bufGenerate];
          POETRY_VIRTUALENVS_IN_PROJECT = true;
          LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [stdenv.cc.cc zlib "/run/opengl-driver"];
          shellHook = ''
            ${lib.getExe poetry} env use ${lib.getExe python}
            ${lib.getExe poetry} install --all-extras --no-root --sync
            set -a
            source .env 2> /dev/null
          '';
        };
      };
    };
}
