{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
  };
  outputs = inputs @ {
    nixpkgs,
    flake-parts,
    systems,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = import systems;
      perSystem = {
        pkgs,
        lib,
        ...
      }: {
        devShells.default = let
          python = pkgs.python311;
          poetry = pkgs.poetry;
        in
          pkgs.mkShell {
            packages = [poetry python];
            POETRY_VIRTUALENVS_IN_PROJECT = true;
            LD_LIBRARY_PATH = lib.makeLibraryPath [pkgs.stdenv.cc.cc];
            shellHook = ''
              ${lib.getExe poetry} env use ${lib.getExe python}
              ${lib.getExe poetry} install --all-extras --no-root
            '';
          };
      };
    };
}
