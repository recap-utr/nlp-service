{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = inputs @ {
    nixpkgs,
    flake-parts,
    systems,
    poetry2nix,
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
        # _module.args.pkgs = import nixpkgs {
        #   inherit system;
        #   config.allowUnfree = true;
        #   config.cudaSupport = true;
        # };
        packages = {
          default = poetry2nix.legacyPackages.${system}.mkPoetryApplication {
            inherit python;
            projectDir = ./.;
            preferWheels = true;
            overrides = poetry2nix.legacyPackages.${system}.overrides.withDefaults (self: super: {
              ml-dtypes = super.ml-dtypes.overridePythonAttrs (old: {
                buildInputs = (old.buildInputs or []) ++ [super.setuptools super.pybind11];
              });
              sentence-transformers = super.sentence-transformers.overridePythonAttrs (old: {
                buildInputs = (old.buildInputs or []) ++ [super.setuptools];
              });
            });
          };
          docker = pkgs.dockerTools.buildLayeredImage {
            name = "nlp-service";
            tag = "latest";
            created = "now";
            config = {
              entrypoint = [(lib.getExe self'.packages.default)];
              cmd = ["--help"];
            };
          };
        };
        devShells.default = pkgs.mkShell {
          packages = [poetry python];
          POETRY_VIRTUALENVS_IN_PROJECT = true;
          LD_LIBRARY_PATH = lib.makeLibraryPath (with pkgs; [
            stdenv.cc.cc
            zlib
          ]);
          shellHook = ''
            ${lib.getExe poetry} env use ${lib.getExe python}
            ${lib.getExe poetry} install --all-extras --no-root
          '';
        };
      };
    };
}
