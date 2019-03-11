source activate dl

maxi="40000"
omega="0.2"
delta="20"
eta="4"

echo "breast_cancer"
for i in {1..5}
do
    python test.py breast_cancer.txt $maxi $omega $delta $eta
done
echo ""

echo "digits"
for i in {1..5}
do
    python test.py digits.txt $maxi $omega $delta $eta
done
echo ""

echo "ionosphere"
for i in {1..5}
do
    python test.py ionosphere.txt $maxi $omega $delta $eta
done
echo ""

echo "mnist_train"
for i in {1..5}
do
    python test.py mnist_train.txt $maxi $omega $delta $eta
done
echo ""

echo "yaleB"
for i in {1..5}
do
    python test.py yale.txt $maxi $omega $delta $eta
done
echo ""

echo "20news"
for i in {1..5}
do
    python test.py 20news.txt $maxi $omega $delta $eta
done
echo ""


