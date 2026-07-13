# Self-contained reconstruction of the benchmark image, rebuilt from the image's layer history
# so it stands alone (no dependency on any private base image).
#
# Stack: Python 3.11.15 + R (RCIT, BLITZ, MXM, bnlearn, ...) + Temurin-21 JRE (Tetrad/FFCI) +
# the suite packages (dagsampler, citests, cbcd, bnmetrics). This is the multi-language
# environment the competitor tests need; GFCM itself needs only the Python layer.
#
# Fully pinned: base image by digest; dagsampler 0.4.0 + citests 0.1.0 + bnmetrics 0.2.2 +
# causalchamber 0.2.8 (PyPI); cbcd @ suite monorepo commit afd7a9d (subdirectory=cbcd);
# py-tetrad @ acd876e (its tetrad-current.jar has the benchmark md5 03ae7e21f1fed311dc93e9782dd684ad).

FROM python:3.11.15-bookworm@sha256:091f0798d4b3b866c8fadb845cca7f96cd18f69b8aa8c05feadd60502fb00a75

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    R_LIBS_USER=/usr/local/lib/R/site-library \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1

# --- system + R build dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg dirmngr \
        build-essential gfortran git \
        libgsl-dev libnlopt-dev libuv1-dev \
        libcurl4-openssl-dev libssl-dev libxml2-dev \
        libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
        libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
        libcairo2-dev libxt-dev libreadline-dev \
        zlib1g-dev libbz2-dev liblzma-dev libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*

# --- R (CRAN bookworm) ---
RUN curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7" \
        | gpg --dearmor -o /usr/share/keyrings/cran-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cran-archive-keyring.gpg] https://cloud.r-project.org/bin/linux/debian bookworm-cran40/" \
        > /etc/apt/sources.list.d/cran.list \
    && apt-get update && apt-get install -y --no-install-recommends r-base r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# --- R packages: RCIT (RCoT), MXM, momentchi2 etc. + C++14 for BLITZ ---
RUN mkdir -p /root/.R && printf 'CXX11STD = -std=gnu++14\nCXX14STD = -std=gnu++14\n' > /root/.R/Makevars \
    && Rscript -e 'install.packages(c("remotes", "BiocManager"), repos="https://cloud.r-project.org/")' \
    && Rscript -e 'install.packages(c("bnlearn","momentchi2","lme4","energy","gsl","Hmisc","visNetwork","fs","htmlwidgets","bslib","quantreg"), repos="https://cloud.r-project.org/")' \
    && Rscript -e 'remotes::install_version("MXM", "1.5.5", repos="https://cloud.r-project.org/", upgrade="never")' \
    && Rscript -e 'remotes::install_github("ericstrobl/RCIT", upgrade="never")' \
    && Rscript -e 'remotes::install_github("ericstrobl/BLITZ", upgrade="never")'

# --- Python scientific stack (pinned as in the benchmark image) ---
RUN pip install --upgrade pip \
    && pip install "numpy<2" \
    && pip install pandas scipy scikit-learn statsmodels \
                   pyarrow joblib tqdm networkx matplotlib \
                   causal-learn rpy2==3.6.4 \
    && pip install tigramite numba \
    && pip install xgboost \
    && pip install causalchamber==0.2.8 \
    && pip install "pycomets @ git+https://github.com/shimenghuang/pycomets.git"

# --- suite packages: dagsampler (PyPI), bnmetrics (PyPI), citests + cbcd (source-only, pin SHAs) ---
RUN pip install "dagsampler==0.4.0" \
    && pip install --no-deps "bnmetrics==0.2.2" \
    && pip install --no-deps "citests==0.1.0" \
    && pip install "cbcd @ git+https://github.com/averinpa/constraint-based-causal-discovery-suite.git@afd7a9d6167c7a1fcc319fd0866beee02685129e#subdirectory=cbcd" \
    && pip install jpype1

# --- Tetrad / FFCI: Temurin 21 JRE (the jar is class version 65) + py-tetrad ---
RUN mkdir -p /etc/apt/keyrings \
    && wget -qO- https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor -o /etc/apt/keyrings/adoptium.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb bookworm main" > /etc/apt/sources.list.d/adoptium.list \
    && apt-get update && apt-get install -y --no-install-recommends temurin-21-jre \
    && rm -rf /var/lib/apt/lists/*
# py-tetrad (bundles tetrad-current.jar, md5 03ae7e21f1fed311dc93e9782dd684ad); pin its SHA
RUN git clone https://github.com/cmu-phil/py-tetrad /opt/py-tetrad \
 && git -C /opt/py-tetrad checkout acd876e87223452709516bc9037a04087e92244b \
 && md5sum /opt/py-tetrad/pytetrad/resources/tetrad-current.jar | grep -q 03ae7e21f1fed311dc93e9782dd684ad

ENV TETRAD_JAR=/opt/py-tetrad/pytetrad/resources/tetrad-current.jar \
    PYTETRAD_DIR=/opt/py-tetrad \
    JAVA_HOME=/usr/lib/jvm/temurin-21-jre-amd64 \
    PYTHONPATH=/opt/py-tetrad:/work

WORKDIR /work

# --- the reproduction code (baked in, so the image is self-contained: no host mount needed).
# Build context is the REPO ROOT: `docker build -f docker/Dockerfile -t gfcm-repro .`
# Kept last so editing code triggers only this cheap layer, not the heavy installs above.
COPY . /work

# --- build-time validation (mirrors the benchmark image's self-check) ---
RUN python3 -c "import dagsampler,citests,jpype,cbcd,bnmetrics; print('built: dagsampler', dagsampler.__version__)" \
    && Rscript -e 'library(BLITZ); library(RCIT); library(quantreg); cat("R pkgs OK\n")' \
    && java -version 2>&1 | head -1 \
    && python3 -c "import jpype; print('JVM ->', jpype.getDefaultJVMPath())"
