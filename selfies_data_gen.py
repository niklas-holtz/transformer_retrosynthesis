import io
import selfies as sf

num_entries = None
path = "data/retrosynthesis-artificial_2.smi"


# Load the lines from the file and separate them
lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
print('Creating dataset for ' + str(num_entries) + ' out of ' + str(len(lines)) + ' found entries of the document.')
# Split the entries
word_pairs = [[w for w in l.split(' >> ')[0:2]] for l in lines[:num_entries]]

"""
# Create selfies dataset
dataset = []
for l in lines[:num_entries]:
    for w in l.split(' >> ')[0:2]:
        encoded = sf.encoder(w)
        dataset.append(encoded)
alphabet = sf.get_alphabet_from_selfies(dataset)
print(alphabet)

alphabet = ['[^]', '.', '[nop]', '[S@expl]', '[/Snexpl]', '[BH3-expl]', '[/B]', '[SiH2expl]', '[\\Cl]', '[/P]', '[=P]', '[Branch1_1]', '[F]',
            '[BH-expl]', '[Expl=Ring1]', '[Branch2_2]', '[/N+expl]', '[P]', '[OH-expl]', '[Ring1]', '[Branch2_3]',
            '[O-expl]', '[N+expl]', '[NH2+expl]', '[=P+expl]', '[#N+expl]', '[SnHexpl]', '[N]', '[=Ptexpl]',
            '[=N+expl]', '[Ring2]', '[C-expl]', '[SHexpl]', '[Expl=Ring2]', '[S]', '[P+expl]', '[B-expl]', '[S-expl]',
            '[Seexpl]', '[=N-expl]', '[=SHexpl]', '[/S]', '[\\C@Hexpl]', '[\\S@@expl]', '[=S]', '[=C]', '[NH-expl]',
            '[C@@Hexpl]', '[I+expl]', '[Branch1_3]', '[Znexpl]', '[C@Hexpl]', '[Cl+3expl]', '[SiHexpl]', '[=N]',
            '[Br-expl]', '[Pdexpl]', '[Cl-expl]', '[\\B]', '[NH+expl]', '[/Br]', '[#N]', '[N-expl]', '[C@expl]', '[/N]',
            '[Mg+expl]', '[Snexpl]', '[Cl]', '[\\S@expl]', '[Mgexpl]', '[PHexpl]', '[/Cl]', '[/F]', '[I]', '[=S@@expl]',
            '[Branch2_1]', '[\\Br]', '[PH4expl]', '[/C@Hexpl]', '[NHexpl]', '[#C]', '[S+expl]', '[Br]', '[Kexpl]',
            '[\\C@@Hexpl]', '[=O]', '[N@+expl]', '[B]', '[\\I]', '[Siexpl]', '[S@@expl]', '[=S+expl]', '[\\O]',
            '[/S@@expl]', '[/O]', '[C]', '[O]', '[Branch1_2]', '[/I]', '[NH3+expl]', '[\\S]', '[Zn+expl]', '[Feexpl]',
            '[\\C]', '[/C@@Hexpl]', '[NH4+expl]', '[/C]', '[Cuexpl]', '[\\F]', '[Liexpl]', '[C@@expl]', '[\\N]',
            '[PH2expl]', '[$]']
            """

alphabet = ['[^]', '.', '[nop]', '[C@expl]', '[Br]', '[/I]', '[B-expl]', '[C@Hexpl]', '[Branch1_2]', '[Expl=Ring2]', '[P]', '[Mg+expl]', 
'[NH2+expl]', '[S@expl]', '[Expl=Ring1]', '[NH+expl]', '[/S]', '[Cl]', '[C@@expl]', '[Seexpl]', '[=SHexpl]', 
'[C@@Hexpl]', '[Branch1_3]', '[SiH2expl]', '[C-expl]', '[=P+expl]', '[P+expl]', '[S@@expl]', '[Siexpl]', '[#C]', 
'[PH4expl]', '[Ring1]', '[SnHexpl]', '[Branch1_1]', '[=P]', '[O-expl]', '[S+expl]', '[BH-expl]', '[Mgexpl]', 
'[/C@Hexpl]', '[\\N]', '[Cuexpl]', '[\\C]', '[=N+expl]', '[N+expl]', '[\\S@expl]', '[F]', '[=N]', '[/S@@expl]', 
'[\\B]', '[Ptexpl]', '[\\F]', '[NHexpl]', '[\\S]', '[PH2expl]', '[Pdexpl]', '[N]', '[/Br]', '[\\S@@expl]', '[=C]', 
'[B]', '[Snexpl]', '[/C]', '[#N+expl]', '[Zn+expl]', '[BH3-expl]', '[#N]', '[Cl+3expl]', '[/Snexpl]', '[/N]', 
'[\\O]', '[I]', '[Branch2_2]', '[/O]', '[\\Cl]', '[#C-expl]', '[/F]', '[/P]', '[C]', '[OH-expl]', '[=O]', '[Feexpl]', 
'[SiHexpl]', '[\\C@Hexpl]', '[NH3+expl]', '[=S@@expl]', '[O]', '[Cl-expl]', '[=Ptexpl]', '[/N+expl]', '[SHexpl]', 
'[Branch2_1]', '[Znexpl]', '[\\Br]', '[PHexpl]', '[\\C@@Hexpl]', '[/B]', '[Liexpl]', '[NH4+expl]', '[/Cl]', 
'[Branch2_3]', '[=S]', '[N-expl]', '[Ring2]', '[I+expl]', '[=N-expl]', '[S]', '[NH-expl]', '[\\I]', '[Kexpl]', 
'[S-expl]', '[/C@@Hexpl]', '[=S+expl]', '[Br-expl]', '[N@+expl]', '[$]'] 

test = "COC(=O)c1ccc(Br)cc1.Nc1ccccc1"
test_set = ['Fc1cc2c(Cl)ncnc2cn1.NC1CCCCCC1',
'n1c(F)cc2c(Cl)ncnc2c1.C1CCC(N)CCC1',
'C1CCC(N)CCC1.c12c(ncnc1Cl)cnc(F)c2',
'c1(F)cc2c(Cl)ncnc2cn1.C1CCC(N)CCC1',
'NC1CCCCCC1.n1cnc2c(c1Cl)cc(F)nc2']

char_to_ix = {s: i for i, s in enumerate(alphabet)}
ix_to_char = {i: s for i, s in enumerate(alphabet)}

for test in test_set:
    print('----')
    print(test)
    t = sf.encoder(test)
    t = '[^]' + t + '[$]'
    print(t)
    t = sf.selfies_to_encoding(t, char_to_ix, enc_type='label')
    #print(t)
    t = sf.encoding_to_selfies(t, ix_to_char, enc_type='label')
    #print(t)
    t = sf.decoder(t)
    print(t)



print("decoding")
k = '[^][C][O][C][Branch1_2][C][=O][C][=C][C][=C][Branch1_1][C][Br][C][=C][Ring1][Branch1_3].[N][C][=C][C][=C][C][=C][Ring1][Branch1_2][$]'
k = sf.decoder(k)
l = '[^][C][O][C][Branch1_2][C][=O][C][=C][C][=C][Branch1_1][C][Br][C][=C][Ring1][Branch1_3].[N][C][=C][C][=C][C][=C][Ring1][Branch1_2][$]'

print(k)