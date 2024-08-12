#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

SMPL Neutral model
echo -e "\nYou need to register at https://smplify.is.tue.mpg.de, assuming you use the same credentials for SMPL, SMPLX, and SMPLify"
read -p "Username (SMPLify):" username
read -p "Password (SMPLify):" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' -O './data/smpl/smplify.zip' --no-check-certificate --continue
unzip data/smpl/smplify.zip -d data/smpl/smplify
mv data/smpl/smplify/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl data/smpl/SMPL_NEUTRAL.pkl
rm -rf data/smpl/smplify
rm -rf data/smpl/smplify.zip

# SMPL Male and Female model
# echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
# read -p "Username (SMPL):" username
# read -p "Password (SMPL):" password
# username=$(urle $username)
# password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' -O './data/smpl/smpl.zip' --no-check-certificate --continue
unzip data/smpl/smpl.zip -d data/smpl/smpl
mv data/smpl/smpl/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl data/smpl/SMPL_FEMALE.pkl
mv data/smpl/smpl/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl data/smpl/SMPL_MALE.pkl
rm -rf data/smpl/smpl
rm -rf data/smpl/smpl.zip

# SMPLX model
# echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
# read -p "Username (SMPLx):" username
# read -p "Password (SMPLx):" password
# username=$(urle $username)
# password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' -O './data/smpl/smplx.zip' --no-check-certificate --continue
unzip data/smpl/smplx.zip -d data/smpl/smplx
mv data/smpl/smplx/models/smplx/SMPLX_FEMALE.pkl data/smpl/SMPLX_FEMLAE.pkl
mv data/smpl/smplx/models/smplx/SMPLX_MALE.pkl data/smpl/SMPLX_MALE.pkl
mv data/smpl/smplx/models/smplx/SMPLX_NEUTRAL.pkl data/smpl/SMPLX_NEUTRAL.pkl
rm -rf data/smpl/smplx
rm -rf data/smpl/smplx.zip
