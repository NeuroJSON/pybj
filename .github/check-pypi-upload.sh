#!/bin/bash
PYPKG_BUILD_VERSION=$(awk -F"-" '{ print $2 }' <<< $(ls dist/ | head -1))
PYPKG_VERSIONS_STRING=$(pip index versions bjdata | grep versions:)
PYPKG_VERSIONS_STRING=${PYPKG_VERSIONS_STRING#*:}
UPLOAD_TO_PYPI=1
while IFS=', ' read -ra PYPKG_VERSIONS_ARRAY; do
  for VERSION in "${PYPKG_VERSIONS_ARRAY[@]}"; do
    if [ "$PYPKG_BUILD_VERSION" = "$VERSION" ]; then
      UPLOAD_TO_PYPI=0
    fi
  done;
done <<< "$PYPKG_VERSIONS_STRING"
if [ "$UPLOAD_TO_PYPI" = 1 ]; then
  echo "Wheel version wasn't found on PyPi.";
else
  echo "Wheel was found on PyPi.";
fi
echo "perform_pypi_upload=$UPLOAD_TO_PYPI" >> $GITHUB_OUTPUT
