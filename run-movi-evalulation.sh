###########################
####### INPUT DEPTH ####### 
###########################

# Radom slot initialization
echo "Loci-s depth movi-c rnd seed = 1"
python -m model.main -cfg configs/movi-c-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 1 > out/eval_random_movi_c_seed_1 2>&1
echo "Loci-s depth movi-c rnd seed = 2"
python -m model.main -cfg configs/movi-c-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 2 > out/eval_random_movi_c_seed_2 2>&1
echo "Loci-s depth movi-c rnd seed = 3"
python -m model.main -cfg configs/movi-c-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 3 > out/eval_random_movi_c_seed_3 2>&1
echo "Loci-s depth movi-c rnd seed = 4"
python -m model.main -cfg configs/movi-c-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 4 > out/eval_random_movi_c_seed_4 2>&1
echo "Loci-s depth movi-c rnd seed = 5"
python -m model.main -cfg configs/movi-c-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 5 > out/eval_random_movi_c_seed_5 2>&1

echo "Loci-s depth movi-d rnd seed = 1"
python -m model.main -cfg configs/movi-d-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 1 > out/eval_random_movi_d_seed_1 2>&1
echo "Loci-s depth movi-d rnd seed = 2"
python -m model.main -cfg configs/movi-d-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 2 > out/eval_random_movi_d_seed_2 2>&1
echo "Loci-s depth movi-d rnd seed = 3"
python -m model.main -cfg configs/movi-d-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 3 > out/eval_random_movi_d_seed_3 2>&1
echo "Loci-s depth movi-d rnd seed = 4"
python -m model.main -cfg configs/movi-d-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 4 > out/eval_random_movi_d_seed_4 2>&1
echo "Loci-s depth movi-d rnd seed = 5"
python -m model.main -cfg configs/movi-d-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 5 > out/eval_random_movi_d_seed_5 2>&1

echo "Loci-s depth movi-e rnd seed = 1"
python -m model.main -cfg configs/movi-e-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 1 > out/eval_random_movi_e_seed_1 2>&1
echo "Loci-s depth movi-e rnd seed = 2"
python -m model.main -cfg configs/movi-e-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 2 > out/eval_random_movi_e_seed_2 2>&1
echo "Loci-s depth movi-e rnd seed = 3"
python -m model.main -cfg configs/movi-e-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 3 > out/eval_random_movi_e_seed_3 2>&1
echo "Loci-s depth movi-e rnd seed = 4"
python -m model.main -cfg configs/movi-e-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 4 > out/eval_random_movi_e_seed_4 2>&1
echo "Loci-s depth movi-e rnd seed = 5"
python -m model.main -cfg configs/movi-e-loci-depth-rnd.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 5 > out/eval_random_movi_e_seed_5 2>&1

# Regularized slot initialization
echo "Loci-s depth movi-c reg seed = 1"
python -m model.main -cfg configs/movi-c-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 1 > out/eval_regularized_movi_c_seed_1 2>&1
echo "Loci-s depth movi-c reg seed = 2"
python -m model.main -cfg configs/movi-c-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 2 > out/eval_regularized_movi_c_seed_2 2>&1
echo "Loci-s depth movi-c reg seed = 3"
python -m model.main -cfg configs/movi-c-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 3 > out/eval_regularized_movi_c_seed_3 2>&1
echo "Loci-s depth movi-c reg seed = 4"
python -m model.main -cfg configs/movi-c-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 4 > out/eval_regularized_movi_c_seed_4 2>&1
echo "Loci-s depth movi-c reg seed = 5"
python -m model.main -cfg configs/movi-c-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 5 > out/eval_regularized_movi_c_seed_5 2>&1

echo "Loci-s depth movi-d reg seed = 1"
python -m model.main -cfg configs/movi-d-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 1 > out/eval_regularized_movi_d_seed_1 2>&1
echo "Loci-s depth movi-d reg seed = 2"
python -m model.main -cfg configs/movi-d-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 2 > out/eval_regularized_movi_d_seed_2 2>&1
echo "Loci-s depth movi-d reg seed = 3"
python -m model.main -cfg configs/movi-d-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 3 > out/eval_regularized_movi_d_seed_3 2>&1
echo "Loci-s depth movi-d reg seed = 4"
python -m model.main -cfg configs/movi-d-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 4 > out/eval_regularized_movi_d_seed_4 2>&1
echo "Loci-s depth movi-d reg seed = 5"
python -m model.main -cfg configs/movi-d-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 5 > out/eval_regularized_movi_d_seed_5 2>&1

echo "Loci-s depth movi-e reg seed = 1"
python -m model.main -cfg configs/movi-e-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 1 > out/eval_regularized_movi_e_seed_1 2>&1
echo "Loci-s depth movi-e reg seed = 2"
python -m model.main -cfg configs/movi-e-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 2 > out/eval_regularized_movi_e_seed_2 2>&1
echo "Loci-s depth movi-e reg seed = 3"
python -m model.main -cfg configs/movi-e-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 3 > out/eval_regularized_movi_e_seed_3 2>&1
echo "Loci-s depth movi-e reg seed = 4"
python -m model.main -cfg configs/movi-e-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 4 > out/eval_regularized_movi_e_seed_4 2>&1
echo "Loci-s depth movi-e reg seed = 5"
python -m model.main -cfg configs/movi-e-loci-depth-reg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 5 > out/eval_regularized_movi_e_seed_5 2>&1

# Segmentation net based slot initialization
echo "Loci-s depth movi-c seg seed = 1"
python -m model.main -cfg configs/movi-c-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 1 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_c_seed_1 2>&1
echo "Loci-s depth movi-c seg seed = 2"
python -m model.main -cfg configs/movi-c-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 2 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_c_seed_2 2>&1
echo "Loci-s depth movi-c seg seed = 3"
python -m model.main -cfg configs/movi-c-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 3 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_c_seed_3 2>&1
echo "Loci-s depth movi-c seg seed = 4"
python -m model.main -cfg configs/movi-c-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 4 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_c_seed_4 2>&1
echo "Loci-s depth movi-c seg seed = 5"
python -m model.main -cfg configs/movi-c-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 5 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_c_seed_5 2>&1

echo "Loci-s depth movi-d seg seed = 1"
python -m model.main -cfg configs/movi-d-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 1 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_d_seed_1 2>&1
echo "Loci-s depth movi-d seg seed = 2"                                                                                                                                        
python -m model.main -cfg configs/movi-d-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 2 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_d_seed_2 2>&1
echo "Loci-s depth movi-d seg seed = 3"                                                                                                                                        
python -m model.main -cfg configs/movi-d-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 3 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_d_seed_3 2>&1
echo "Loci-s depth movi-d seg seed = 4"                                                                                                                                        
python -m model.main -cfg configs/movi-d-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 4 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_d_seed_4 2>&1
echo "Loci-s depth movi-d seg seed = 5"                                                                                                                                        
python -m model.main -cfg configs/movi-d-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 5 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_d_seed_5 2>&1

echo "Loci-s depth movi-e seg seed = 1"
python -m model.main -cfg configs/movi-e-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 1 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_e_seed_1 2>&1
echo "Loci-s depth movi-e seg seed = 2"                                                                                                                                        
python -m model.main -cfg configs/movi-e-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 2 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_e_seed_2 2>&1
echo "Loci-s depth movi-e seg seed = 3"                                                                                                                                        
python -m model.main -cfg configs/movi-e-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 3 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_e_seed_3 2>&1
echo "Loci-s depth movi-e seg seed = 4"                                                                                                                                        
python -m model.main -cfg configs/movi-e-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 4 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_e_seed_4 2>&1
echo "Loci-s depth movi-e seg seed = 5"                                                                                                                                        
python -m model.main -cfg configs/movi-e-loci-depth-seg.json -validate -single-gpu --load checkpoints/Loci-s-depth.ckpt --seed 5 --load-proposal checkpoints/ProposalDepth.ckpt > out/eval_segmentation_movi_e_seed_5 2>&1


##############################
####### NO INPUT DEPTH #######
##############################

# Radom slot initialization
echo "Loci-s movi-e rnd seed = 1"
python -m model.main -cfg configs/movi-e-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 1 > out/eval_no_depth_random_movi_e_seed_1 2>&1
echo "Loci-s movi-e rnd seed = 2"
python -m model.main -cfg configs/movi-e-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 2 > out/eval_no_depth_random_movi_e_seed_2 2>&1
echo "Loci-s movi-e rnd seed = 3"
python -m model.main -cfg configs/movi-e-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 3 > out/eval_no_depth_random_movi_e_seed_3 2>&1
echo "Loci-s movi-e rnd seed = 4"
python -m model.main -cfg configs/movi-e-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 4 > out/eval_no_depth_random_movi_e_seed_4 2>&1
echo "Loci-s movi-e rnd seed = 5"
python -m model.main -cfg configs/movi-e-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 5 > out/eval_no_depth_random_movi_e_seed_5 2>&1

echo "Loci-s movi-d rnd seed = 1"
python -m model.main -cfg configs/movi-d-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 1 > out/eval_no_depth_random_movi_d_seed_1 2>&1
echo "Loci-s movi-d rnd seed = 2"
python -m model.main -cfg configs/movi-d-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 2 > out/eval_no_depth_random_movi_d_seed_2 2>&1
echo "Loci-s movi-d rnd seed = 3"
python -m model.main -cfg configs/movi-d-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 3 > out/eval_no_depth_random_movi_d_seed_3 2>&1
echo "Loci-s movi-d rnd seed = 4"
python -m model.main -cfg configs/movi-d-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 4 > out/eval_no_depth_random_movi_d_seed_4 2>&1
echo "Loci-s movi-d rnd seed = 5"
python -m model.main -cfg configs/movi-d-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 5 > out/eval_no_depth_random_movi_d_seed_5 2>&1

echo "Loci-s movi-c rnd seed = 1"
python -m model.main -cfg configs/movi-c-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 1 > out/eval_no_depth_random_movi_c_seed_1 2>&1
echo "Loci-s movi-c rnd seed = 2"
python -m model.main -cfg configs/movi-c-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 2 > out/eval_no_depth_random_movi_c_seed_2 2>&1
echo "Loci-s movi-c rnd seed = 3"
python -m model.main -cfg configs/movi-c-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 3 > out/eval_no_depth_random_movi_c_seed_3 2>&1
echo "Loci-s movi-c rnd seed = 4"
python -m model.main -cfg configs/movi-c-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 4 > out/eval_no_depth_random_movi_c_seed_4 2>&1
echo "Loci-s movi-c rnd seed = 5"
python -m model.main -cfg configs/movi-c-loci-rnd.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 5 > out/eval_no_depth_random_movi_c_seed_5 2>&1

# Regularized slot initialization
echo "Loci-s movi-e reg seed = 1"
python -m model.main -cfg configs/movi-e-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 1 > out/eval_no_depth_regularized_movi_e_seed_1 2>&1
echo "Loci-s movi-e reg seed = 2"
python -m model.main -cfg configs/movi-e-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 2 > out/eval_no_depth_regularized_movi_e_seed_2 2>&1
echo "Loci-s movi-e reg seed = 3"
python -m model.main -cfg configs/movi-e-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 3 > out/eval_no_depth_regularized_movi_e_seed_3 2>&1
echo "Loci-s movi-e reg seed = 4"
python -m model.main -cfg configs/movi-e-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 4 > out/eval_no_depth_regularized_movi_e_seed_4 2>&1
echo "Loci-s movi-e reg seed = 5"
python -m model.main -cfg configs/movi-e-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 5 > out/eval_no_depth_regularized_movi_e_seed_5 2>&1
                                               
echo "Loci-s movi-d reg seed = 1"
python -m model.main -cfg configs/movi-d-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 1 > out/eval_no_depth_regularized_movi_d_seed_1 2>&1
echo "Loci-s movi-d reg seed = 2"
python -m model.main -cfg configs/movi-d-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 2 > out/eval_no_depth_regularized_movi_d_seed_2 2>&1
echo "Loci-s movi-d reg seed = 3"
python -m model.main -cfg configs/movi-d-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 3 > out/eval_no_depth_regularized_movi_d_seed_3 2>&1
echo "Loci-s movi-d reg seed = 4"
python -m model.main -cfg configs/movi-d-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 4 > out/eval_no_depth_regularized_movi_d_seed_4 2>&1
echo "Loci-s movi-d reg seed = 5"
python -m model.main -cfg configs/movi-d-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 5 > out/eval_no_depth_regularized_movi_d_seed_5 2>&1
                                               
echo "Loci-s movi-c reg seed = 1"
python -m model.main -cfg configs/movi-c-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 1 > out/eval_no_depth_regularized_movi_c_seed_1 2>&1
echo "Loci-s movi-c reg seed = 2"
python -m model.main -cfg configs/movi-c-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 2 > out/eval_no_depth_regularized_movi_c_seed_2 2>&1
echo "Loci-s movi-c reg seed = 3"
python -m model.main -cfg configs/movi-c-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 3 > out/eval_no_depth_regularized_movi_c_seed_3 2>&1
echo "Loci-s movi-c reg seed = 4"
python -m model.main -cfg configs/movi-c-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 4 > out/eval_no_depth_regularized_movi_c_seed_4 2>&1
echo "Loci-s movi-c reg seed = 5"
python -m model.main -cfg configs/movi-c-loci-reg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 5 > out/eval_no_depth_regularized_movi_c_seed_5 2>&1

# Segmentation net based slot initialization
echo "Loci-s movi-e seg seed = 1"
python -m model.main -cfg configs/movi-e-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 1 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_e_seed_1 2>&1
echo "Loci-s movi-e seg seed = 2"                                                                                                                                  
python -m model.main -cfg configs/movi-e-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 2 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_e_seed_2 2>&1
echo "Loci-s movi-e seg seed = 3"                                                                                                                                  
python -m model.main -cfg configs/movi-e-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 3 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_e_seed_3 2>&1
echo "Loci-s movi-e seg seed = 4"                                                                                                                                  
python -m model.main -cfg configs/movi-e-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 4 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_e_seed_4 2>&1
echo "Loci-s movi-e seg seed = 5"                                                                                                                                  
python -m model.main -cfg configs/movi-e-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 5 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_e_seed_5 2>&1
                                               
echo "Loci-s movi-d seg seed = 1"
python -m model.main -cfg configs/movi-d-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 1 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_d_seed_1 2>&1
echo "Loci-s movi-d seg seed = 2"                                                                                                                             
python -m model.main -cfg configs/movi-d-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 2 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_d_seed_2 2>&1
echo "Loci-s movi-d seg seed = 3"                                                                                                                             
python -m model.main -cfg configs/movi-d-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 3 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_d_seed_3 2>&1
echo "Loci-s movi-d seg seed = 4"                                                                                                                             
python -m model.main -cfg configs/movi-d-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 4 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_d_seed_4 2>&1
echo "Loci-s movi-d seg seed = 5"                                                                                                                             
python -m model.main -cfg configs/movi-d-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 5 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_d_seed_5 2>&1
                                               
echo "Loci-s movi-c seg seed = 1"
python -m model.main -cfg configs/movi-c-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 1 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_c_seed_1 2>&1
echo "Loci-s movi-c seg seed = 2"                                                                                                                             
python -m model.main -cfg configs/movi-c-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 2 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_c_seed_2 2>&1
echo "Loci-s movi-c seg seed = 3"                                                                                                                             
python -m model.main -cfg configs/movi-c-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 3 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_c_seed_3 2>&1
echo "Loci-s movi-c seg seed = 4"                                                                                                                             
python -m model.main -cfg configs/movi-c-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 4 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_c_seed_4 2>&1
echo "Loci-s movi-c seg seed = 5"                                                                                                                             
python -m model.main -cfg configs/movi-c-loci-seg.json -validate -single-gpu --load checkpoints/Loci-s.ckpt --seed 5 --load-proposal checkpoints/Proposal.ckpt > out/eval_no_depth_segmentation_movi_c_seed_5 2>&1
