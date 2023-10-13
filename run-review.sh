##############################
####### GENERALIZATION #######
##############################

# random slot initialization
echo "Loci-s review generalization rnd seed = 1"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-random.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 1 > out/eval_review_generalization_random_seed_1 2>&1 
echo "Loci-s review generalization rnd seed = 2"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-random.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 2 > out/eval_review_generalization_random_seed_2 2>&1 
echo "Loci-s review generalization rnd seed = 3"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-random.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 3 > out/eval_review_generalization_random_seed_3 2>&1
echo "Loci-s review generalization rnd seed = 4"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-random.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 4 > out/eval_review_generalization_random_seed_4 2>&1 
echo "Loci-s review generalization rnd seed = 5"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-random.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 5 > out/eval_review_generalization_random_seed_5 2>&1

# regularized slot initialization
echo "Loci-s review generalization reg seed = 1"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-regularization.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 1 > out/eval_review_generalization_regularized_seed_1 2>&1 
echo "Loci-s review generalization reg seed = 2"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-regularization.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 2 > out/eval_review_generalization_regularized_seed_2 2>&1 
echo "Loci-s review generalization reg seed = 3"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-regularization.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 3 > out/eval_review_generalization_regularized_seed_3 2>&1
echo "Loci-s review generalization reg seed = 4"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-regularization.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 4 > out/eval_review_generalization_regularized_seed_4 2>&1 
echo "Loci-s review generalization reg seed = 5"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-regularization.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 5 > out/eval_review_generalization_regularized_seed_5 2>&1

# segmentation net based slot initialization
echo "Loci-s review generalization seg seed = 1"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-segmentation.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 1 --load-proposal checkpoints/ProposalReview.ckpt > out/eval_review_generalization_segmentation_seed_1 2>&1 
echo "Loci-s review generalization seg seed = 2"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-segmentation.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 2 --load-proposal checkpoints/ProposalReview.ckpt > out/eval_review_generalization_segmentation_seed_2 2>&1 
echo "Loci-s review generalization seg seed = 3"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-segmentation.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 3 --load-proposal checkpoints/ProposalReview.ckpt > out/eval_review_generalization_segmentation_seed_3 2>&1
echo "Loci-s review generalization seg seed = 4"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-segmentation.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 4 --load-proposal checkpoints/ProposalReview.ckpt > out/eval_review_generalization_segmentation_seed_4 2>&1 
echo "Loci-s review generalization seg seed = 5"
python -m model.main -cfg configs/slot-attention-review-loci-generalization-segmentation.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 5 --load-proposal checkpoints/ProposalReview.ckpt > out/eval_review_generalization_segmentation_seed_5 2>&1

########################
####### TEST SET #######
########################

# random slot initialization
echo "Loci-s review test rnd seed = 1"
python -m model.main -cfg configs/slot-attention-review-loci-test-random.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 1 > out/eval_review_test_random_seed_1 2>&1 
echo "Loci-s review test rnd seed = 2"
python -m model.main -cfg configs/slot-attention-review-loci-test-random.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 2 > out/eval_review_test_random_seed_2 2>&1 
echo "Loci-s review test rnd seed = 3"
python -m model.main -cfg configs/slot-attention-review-loci-test-random.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 3 > out/eval_review_test_random_seed_3 2>&1 
echo "Loci-s review test rnd seed = 4"
python -m model.main -cfg configs/slot-attention-review-loci-test-random.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 4 > out/eval_review_test_random_seed_4 2>&1
echo "Loci-s review test rnd seed = 5"
python -m model.main -cfg configs/slot-attention-review-loci-test-random.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 5 > out/eval_review_test_random_seed_5 2>&1 

# regularized slot initialization
echo "Loci-s review test reg seed = 1"
python -m model.main -cfg configs/slot-attention-review-loci-test-regularization.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 1 > out/eval_review_test_regularized_seed_1 2>&1 
echo "Loci-s review test reg seed = 2"
python -m model.main -cfg configs/slot-attention-review-loci-test-regularization.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 2 > out/eval_review_test_regularized_seed_2 2>&1 
echo "Loci-s review test reg seed = 3"
python -m model.main -cfg configs/slot-attention-review-loci-test-regularization.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 3 > out/eval_review_test_regularized_seed_3 2>&1 
echo "Loci-s review test reg seed = 4"
python -m model.main -cfg configs/slot-attention-review-loci-test-regularization.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 4 > out/eval_review_test_regularized_seed_4 2>&1
echo "Loci-s review test reg seed = 5"
python -m model.main -cfg configs/slot-attention-review-loci-test-regularization.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 5 > out/eval_review_test_regularized_seed_5 2>&1 

# segmentation net based slot initialization
echo "Loci-s review test seg seed = 1"
python -m model.main -cfg configs/slot-attention-review-loci-test-segmentation.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 1 --load-proposal checkpoints/ProposalReview.ckpt > out/eval_review_test_segmentation_seed_1 2>&1 
echo "Loci-s review test seg seed = 2"
python -m model.main -cfg configs/slot-attention-review-loci-test-segmentation.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 2 --load-proposal checkpoints/ProposalReview.ckpt > out/eval_review_test_segmentation_seed_2 2>&1 
echo "Loci-s review test seg seed = 3"
python -m model.main -cfg configs/slot-attention-review-loci-test-segmentation.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 3 --load-proposal checkpoints/ProposalReview.ckpt > out/eval_review_test_segmentation_seed_3 2>&1 
echo "Loci-s review test seg seed = 4"
python -m model.main -cfg configs/slot-attention-review-loci-test-segmentation.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 4 --load-proposal checkpoints/ProposalReview.ckpt > out/eval_review_test_segmentation_seed_4 2>&1
echo "Loci-s review test seg seed = 5"
python -m model.main -cfg configs/slot-attention-review-loci-test-segmentation.json -validate -single-gpu --load checkpoints/Loci-s-review.ckpt --seed 5 --load-proposal checkpoints/ProposalReview.ckpt > out/eval_review_test_segmentation_seed_5 2>&1 
