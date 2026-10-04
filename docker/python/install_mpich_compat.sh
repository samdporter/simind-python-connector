#!/bin/sh
# SIMIND's Linux binary links against the historical MPICH names
# libmpi.so.12 / libmpifort.so.12. Debian's mpich package ships
# libmpich.so.12 / libmpichfort.so.12; expose the expected names when the
# loader cannot find them.
set -eu

multiarch="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("MULTIARCH"))')"
libdir="/usr/lib/${multiarch}"

link_if_missing() {
    target="$1"
    link_name="$2"
    if [ ! -e "${libdir}/${link_name}" ] && [ -e "${libdir}/${target}" ]; then
        ln -s "${target}" "${libdir}/${link_name}"
    fi
}

link_if_missing libmpich.so.12 libmpi.so.12
link_if_missing libmpichfort.so.12 libmpifort.so.12

ldconfig
