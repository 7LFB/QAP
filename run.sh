# 28 areas,29 eccs, 30 circus, 31 intens,  32 entropys, 33 shapes, 34 extents, 35 perimeters, 36 percents 

model_name='QAP'
xprompt=1
num_classes=4
data_version=0
logdir='XPrompt'
generate_props=0
prior_nuclei=0
prior_white=0
dilate=0
convex_hull=0
clear_border=0


for seed in 1024
do
for mversion in 36
do
for augment in 0
do
for auto_augment in 0
do
for augment_index in 'none'
do
for se_kernel in 11
do
for prompt_index_str_s in '0-1-2-3'
do
for prompt_index_str_m in '0-1-7'
do
for kfunc_version in 'S0V0'
do
for area_thd in 0
do
for ratio in 1.0
do 
for fold in 0 1 2 3 4
do
CUDA_VISIBLE_DEVICES=3 python train.py\
    --model_name "${model_name}-DataV${data_version}-Ratio${ratio}-AugIndex${augment_index}AutoAug${auto_augment}-promptIndexS${prompt_index_str_s}M${prompt_index_str_m}-kfunc${kfunc_version}-D${dilate}S${se_kernel}T${area_thd}H${convex_hull}B${clear_border}"\
	--fold $fold\
    --mversion $mversion\
    --prompt_index_str_s $prompt_index_str_s\
    --prompt_index_str_m $prompt_index_str_m\
    --kfunc_version $kfunc_version\
    --xprompt $xprompt\
    --num_classes $num_classes\
    --ratio $ratio\
    --augment $augment\
    --auto_augment $auto_augment\
    --augment_index $augment_index\
    --data_version $data_version\
    --seed $seed\
    --logdir $logdir\
    --generate_props $generate_props\
    --prior_nuclei $prior_nuclei\
    --prior_white $prior_white\
    --se_kernel $se_kernel\
    --dilate $dilate\
    --area_thd $area_thd\
    --convex_hull $convex_hull\
    --clear_border $clear_border
done
done
done
done
done
done
done
done
done
done
done
done