for i in 0.03
do
    python3 feature_generation_vae.py -budget $i > output_feat_$i.txt
    python3 main_new.py -budget $i -target 0 > output_main_$i.txt
done 
