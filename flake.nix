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
      url = "github:mirkolenz/flocken/v1";
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
        python = pkgs.python310;
        poetry = pkgs.poetry;
      in {
        _module.args.pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
        apps.dockerManifest = {
          type = "app";
          program = lib.getExe (flocken.legacyPackages.${system}.mkDockerManifest {
            branch = builtins.getEnv "GITHUB_REF_NAME";
            name = "ghcr.io/" + builtins.getEnv "GITHUB_REPOSITORY";
            version = builtins.getEnv "VERSION";
            images = with self.packages; [x86_64-linux.docker];
          });
        };
        packages = {
          default = poetry2nix.legacyPackages.${system}.mkPoetryApplication {
            inherit python;
            projectDir = ./.;
            preferWheels = true;
            overrides = poetry2nix.legacyPackages.${system}.overrides.withDefaults (self: super: {
              sentence-transformers = super.sentence-transformers.overridePythonAttrs (old: {
                buildInputs = (old.buildInputs or []) ++ [super.setuptools];
              });
            });
          };
          docker = pkgs.dockerTools.buildImage {
            name = "nlp-service";
            tag = "latest";
            created = "now";
            config = {
              entrypoint = [(lib.getExe self'.packages.default)];
              cmd = ["0.0.0.0:50100"];
            };
          };
          releaseEnv = pkgs.buildEnv {
            name = "release-env";
            paths = [poetry python];
          };
        };
        devShells.default = pkgs.mkShell {
          packages = [poetry python];
          POETRY_VIRTUALENVS_IN_PROJECT = true;
          LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [stdenv.cc.cc zlib];
          LD_PRELOAD = "/run/opengl-driver/lib/libcuda.so";
          shellHook = ''
            ${lib.getExe poetry} env use ${lib.getExe python}
            ${lib.getExe poetry} install --all-extras --no-root
          '';
        };
      };
    };
}
