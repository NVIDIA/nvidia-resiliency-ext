
## Nvidia Resiliency Extension (NVRx) OSS Contribution Rules

Thank you for your interest in contributing to this project. We value collaboration, clear ownership, and high standards of responsibility. These guidelines outline how to contribute effectively and responsibly, ensuring that code quality, project direction, and accountability are maintained. 

Please note that as a contributor, you are responsible for responding to feedback and ongoing maintenance of your contributions. By following these guidelines, you help maintain a high-quality, collaborative and productive environment for all contributors. Any code, comments, issues and other contributions violating these guidelines will be subject to removal.  

#### Issue Tracking

* All enhancement, bugfix, or change requests must begin with the creation of a [NVRx Issue Request](TBD).
  * The issue request must be reviewed by NVRx engineers and approved prior to code review.


#### Coding Guidelines

- All source code contributions must follow the existing conventions in the relevant file, submodule, module, and project when you add new code or when you extend/fix existing functionality.

- Avoid introducing unnecessary complexity into existing code so that maintainability and readability are preserved.

- Try to keep pull requests (PRs) as concise as possible:
  - Avoid committing commented-out code.
  - Wherever possible, each PR should address a single concern. If there are several otherwise-unrelated things that should be fixed to reach a desired endpoint, our recommendation is to open several PRs and indicate the dependencies in the description. The more complex the changes are in a single PR, the more time it will take to review those changes.

- To ensure code consistency and maintainability across the project, please format and lint your code using the following tools before committing any changes:
  - We use black to automatically format Python code. It enforces a consistent style by reformatting code according to a set of rules.
  - To format your code, run:
```
black .
```
  - isort is used to sort and format import statements automatically. Ensure that your imports are ordered correctly by running:
```
isort .
```
  - ruff is a fast Python linter that helps catch common issues. Please run ruff to check for and fix linting problems:
```
ruff check .
```

- Write commit titles using imperative mood and [these rules](https://chris.beams.io/posts/git-commit/), and reference the Issue number corresponding to the PR. Following is the recommended format for commit texts:
```
#<Issue Number> - <Commit Title>

<Commit Body>
```

- Ensure that the build log is clean, meaning no warnings or errors should be present.

- Ensure that all unit tests pass prior to submitting your code.

- All OSS components must contain accompanying documentation (READMEs) describing the functionality, dependencies, and known issues.

  - See `README.md` for existing samples and plugins for reference.

- All OSS components must have an accompanying test.

  - If introducing a new component, such as a plugin, provide a test sample to verify the functionality.

- Make sure that you can contribute your work to open source (no license and/or patent conflict is introduced by your code). You will need to [`sign`](#signing-your-work) your commit.

- Thanks in advance for your patience as we review your contributions; we do appreciate them!


#### Pull Requests
Developer workflow for code contributions is as follows:

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the [upstream](TBD) NVRx OSS repository.

2. Git clone the forked repository and push changes to the personal fork.

  ```bash
git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git NVRx 
# Checkout the targeted branch and commit changes
# Push the commits to a branch on the fork (remote).
git push -u origin <local-branch>:<remote-branch>
  ```

3. Once the code changes are staged on the fork and ready for review, a [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) can be [requested](https://help.github.com/en/articles/creating-a-pull-request) to merge the changes from a branch of the fork into a selected branch of upstream.
  * Exercise caution when selecting the source and target branches for the PR.
    Note that versioned releases of NVRx OSS are posted to `release/` branches of the upstream repo.
  * Creation of a PR creation kicks off the code review process.
  * Atleast one NVRx engineer will be assigned for the review.
  * While under review, mark your PRs as work-in-progress by prefixing the PR title with [WIP].

4. With CI/CD process in place, the PR will be accepted and the corresponding issue closed only after adequate testing has been completed, manually, by the developer and NVRx engineer reviewing the code.

#### Documentation Building

When contributing documentation changes, ensure the documentation builds correctly. See the [docs CI workflow](https://github.com/NVIDIA/nvidia-resiliency-ext/blob/main/.github/workflows/build_docs.yml) for up-to-date instructions:

   ```bash
   pip install -U sphinx sphinx-rtd-theme sphinxcontrib-napoleon sphinx_copybutton lightning psutil defusedxml
   sphinx-build -b html docs/source public/

   # alternatively,
   cd docs
   make html
   ```
   You can then view the locally built documentation under `public` directory or `docs/build/html` (e.g., `open public/index.html`). Ensure that all documentation changes are properly formatted and that the build completes without warnings or errors.

#### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.
    
* To sign your Git commits, follow these steps:
  1. Generate a GPG Key (if you don’t have one). Run the following command to generate a new GPG key:
     
     ```bash
     $ gpg --full-generate-key
     ```
     * Select: RSA and RSA (default)
     * Choose a key size: 4096
     * Set an expiration date (or use default: no expiration)
     * Provide your full name and email (must match your Git settings)
  
  3. After generation, list your keys. Find the GPG key ID (it looks like ABCD1234EFGH5678).
     
     ```bash
     $ gpg --list-secret-keys --keyid-format=long
     ```
     
  5. Configure Git to Use Your GPG Key. Run the following commands that tell Git to sign all commits by default.
     
     ```bash
     $ git config --global user.signingkey ABCD1234EFGH5678
     $ git config --global commit.gpgsign true
     ```
     
  7. Export and add the GPG Key to GitHub/GitLab.
     
     * Export the public key:
     ```bash
     $ gpg --armor --export ABCD1234EFGH5678
     ```
     
     * Copy the output and add it to your GitHub/GitLab under:
       * GitHub: Settings → SSH and GPG keys → New GPG Key
       * GitLab: Preferences → GPG Keys

* Now, to sign off on a commit you simply use the `--gpg-sign` (or `-S`) option when committing your changes:
  
  ```bash
  $ git commit -S -m "Add cool feature."
  ```
  
  This will sign the commit message, which can be verified by running:

  ```bash
  $ git log --show-signature"
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
