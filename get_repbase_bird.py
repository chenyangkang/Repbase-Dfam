name_list=open("name.1_531","r").read()
name_list=name_list.split("\n")
genus_list=[]
for i in name_list:
    genus=i.split("_")[0]
    genus_list.append(genus)

name_list.extend(genus_list)
name_list.append("Aves")
# print(name_list)

name_list=",".join(name_list)

repbase=open("Dfam_3_4_curatedonly.fasta","r").read()
repbase=repbase.split(">")[1:]

outfile=open("repbase_bird_dfam_curatedonly.fasta","w")
for i in repbase:
    s=i.split("\n")
    # print(s)
    cu_header=s[0].split(" ")[1].strip("@")
    if cu_header in name_list:  
        print(cu_header)  
        header=[s[0].split(" ")[0]]
        body=s[1:]
        # print(body)
        header.extend(body)
        res="\n".join(header)
        # print(res)
        outfile.write(">%s"%(res))
outfile.close()




