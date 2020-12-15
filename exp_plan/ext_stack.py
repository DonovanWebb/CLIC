import mrcfile as mrc

with mrc.open('7_5_projs.mrcs') as f:
    data = f.data
print(data.shape)
    
num = 0
for count in range(700):
    im = data[count]
    mrc.new(f'7_5_projs/{num}.mrc', im)
    num += 1

