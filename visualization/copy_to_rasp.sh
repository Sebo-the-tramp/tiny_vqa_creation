# copy index.html
SRC="/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/visualization/"
DST="/home/cavadahub/scsv/dataset-server/"

rsync -av $SRC \
     --rsync-path="sudo rsync --mkpath" \
     cavadahub@192.168.0.30:$DST

# copy the results
SRC="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_22_general"
DST="/home/cavadahub/scsv/dataset-server/output/"

rsync -av $SRC \
    --rsync-path="sudo rsync --mkpath" \
    cavadahub@192.168.0.30:$DST
