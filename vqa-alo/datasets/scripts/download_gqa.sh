mkdir -p data/gqaood
cd data/gqaood
wget -c https://nlp.stanford.edu/data/gqa/spatialFeatures.zip
wget -c https://nlp.stanford.edu/data/gqa/objectFeatures.zip

unzip spatialFeatures.zip
python ../preproc_gqa_feat.py --mode=spatial --spatial_dir=./spatialFeatures --out_dir=./feats/gqa-grid
rm -r spatialFeatures.zip ./spatialFeatures

unzip objectFeatures.zip
python ../preproc_gqa_feat.py --mode=object --object_dir=./objectFeatures --out_dir=./feats/gqa-frcn
rm -r objectFeatures.zip ./objectFeatures

git clone https://github.com/gqa-ood/GQA-OOD.git
unzip GQA-OOD
mv GQA-OOD/data/* ./