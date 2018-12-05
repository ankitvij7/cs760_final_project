#!/bin/bash

SAMPLE_RATE=22050

# fetch_clip(videoID, startTime, endTime)
fetch_clip() {
  echo "Fetching $1 ..."
  outname="$1_$2"
  echo "Building $outname"
  if [ -f "wav_outputs/${outname}.wav.gz" ]; then
    echo "Already have it."
    return
  fi

  youtube-dl https://youtube.com/watch?v=$1 \
    --quiet --extract-audio --audio-format wav \
    --output "$outname.%(ext)s"
  if [ $? -eq 0 ]; then
    # If we don't pipe `yes`, ffmpeg seems to steal a
    # character from stdin. I have no idea why.
    echo "Extracting $2 to $3 ..."
    yes | ffmpeg -loglevel quiet -i "./$outname.wav" -ar $SAMPLE_RATE \
      -ss "$2" -to "$3" "./${outname}_out.wav"
    rm -f "./$outname.wav"
    mv "./${outname}_out.wav" "./$outname.wav"
    echo "Compressing $outname.wav ..."
    gzip "./$outname.wav"
    mv "./$outname.wav.gz" "./wav_outputs/$outname.wav.gz"
  else
    # Give the user a chance to Ctrl+C.
    sleep 1
  fi
}

grep -E '^[^#]' | while read line
do
  fetch_clip $(echo "$line" | sed -E 's/, / /g')
done