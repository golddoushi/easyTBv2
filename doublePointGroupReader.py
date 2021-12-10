import numpy as np
import re

class PointGroupOperator:
    def __init__(self,id,SeitzSymbol,rotation_r,rotation_s,class_index,irrepList,characterList) -> None:
        self.id=id
        self.SeitzSymbol=SeitzSymbol
        self.rotation_r=rotation_r
        self.rotation_s=rotation_s
        self.class_index=class_index
        self.irrepList=irrepList
        self.characterVec=characterList

class PointGroupClass:
    def __init__(self,id,operatorList,characterList) -> None:
        self.id=id
        self.operatorList=operatorList
        self.characterVec=characterList

def evalMathExpression(_inStr):
    inStr=str.strip(_inStr)
    #print('instr %s'%inStr)
    
    # check double divider
    divideList=re.findall("[/]",inStr)
    if len(divideList)==2:
        newExp1,newExp2=inStr.split('√')
        return evalMathExpression(newExp1)*evalMathExpression('√'+newExp2)

    # check exp
    str_list=list(inStr)
    first_ele=str_list.pop(0)
    if first_ele=='e':
        newExp=''.join(str_list)
        return np.exp(evalMathExpression(newExp))
    
    # check divider, only one divider expected
    divideList=re.findall("[/]",inStr)
    if len(divideList)==1:
        numerator, denominator= [x for x in inStr.split("/") if x!='']
        #value1=evalMathExpression(numerator)
        #value2=evalMathExpression(denominator)
        #print(value1,'divide by',value2)
        return evalMathExpression(numerator)/evalMathExpression(denominator)
    
    str_list=list(inStr)
    if len(str_list)==1:
        if str_list[0]=='i': return 1j
        if str_list[0]=='π': return np.pi
        return int(inStr)
    # check global sign, tuples
    str_list=list(inStr)
    #print(str_list)#,str_list[0])#,str_list[1])#,str_list[len(str_list)-1])
    if str_list[0]=='-' and str_list[1]=='(' and str_list[-1]==')':
        str_list.pop(0)
        str_list.pop(0)
        str_list.pop()
        newExp=''.join(str_list)
        return -1*evalMathExpression(newExp)
    elif str_list[0]=='-' and str_list[1]=='(':
        # find other tuple
        str_list.pop(0)
        str_list.pop(0)
        str_sub_list=[]
        while(True):
            ele=str_list.pop(0)
            if ele==')':break
            str_sub_list.append(ele)
        newExp1=''.join(str_sub_list)
        newExp2=''.join(str_list)
        return -1*evalMathExpression(newExp1)*evalMathExpression(newExp2)
    elif str_list[0]=='(' and str_list[-1]==')':
        #print('here')
        str_list.pop(0)
        str_list.pop()
        #print(str_list)
        newExp=''.join(str_list)
        return evalMathExpression(newExp)
    elif str_list[0]=='(':
        str_list.pop(0)
        str_sub_list=[]
        while(True):
            ele=str_list.pop(0)
            if ele==')':break
            str_sub_list.append(ele)
        newExp1=''.join(str_sub_list)
        newExp2=''.join(str_list)
        return evalMathExpression(newExp1)*evalMathExpression(newExp2)
    
    # check + - 
    plusMinusList=re.findall("[+-]",inStr)
    if len(plusMinusList)==1:
        operator=plusMinusList[0]
        if operator=='+':
            sumList=[x for x in inStr.split("+") if x!='']
            if len(sumList)==2:
                return evalMathExpression(sumList[0])+evalMathExpression(sumList[1])
            else:
                print("Error, cannot find two parts to perform add")
                print(inStr)
                exit()
        elif operator=='-':
            substractList=[x for x in inStr.split("-") if x!='']
            if len(substractList)==2:
                return evalMathExpression(substractList[0])-evalMathExpression(substractList[1])
            elif len(substractList)==1:
                return -1*evalMathExpression(substractList[0])
            else:
                print("Error, unexpected sub-expression")
                print(inStr)
                exit()
    elif len(plusMinusList)==2:
        assert plusMinusList[0]=='-', 'found two +- in %s, the first sign must be -'%inStr
        splitOperator=plusMinusList[1]
        str_list=list(inStr)
        str_list.pop(0)
        newExp=''.join(str_list)
        part1, part2= newExp.split(splitOperator)
        if splitOperator=='+':
            return -evalMathExpression(part1)+evalMathExpression(part2)
        elif splitOperator=='-':
            return -evalMathExpression(part1)-evalMathExpression(part2)
        else:
            print('Error, unexpected sub-expression, after found 2 +/- operators')
            print(inStr)
            exit()
    
    # check i
    sign_i=re.findall("[i]",inStr)
    #print(sign_i)
    if len(sign_i)==1:
        sub_str=[x for x in inStr.split('i') if x!='']
        #print(sub_str)
        if len(sub_str)==1:
            return 1j*evalMathExpression(sub_str[0])
        elif len(sub_str)==2:
            return 1j*evalMathExpression(sub_str[0])*evalMathExpression(sub_str[1])
        else:
            print('Error, unexpected sub-expression')
            print(inStr)
            exit()
    elif len(sign_i)>1:
        print('Error, unexpected sub-expression, found %d i'%len(sign_i))
        print(inStr)
        exit()
    
    # check pi
    sign_pi=re.findall("[π]",inStr)
    if len(sign_pi)==1:
        sub_str=[x for x in inStr.split('π') if x!='']
        #print(sub_str)
        if len(sub_str)==1:
            return np.pi*evalMathExpression(sub_str[0])
        elif len(sub_str)==2:
            return np.pi*evalMathExpression(sub_str[0])*evalMathExpression(sub_str[1])
        else:
            print('Error, unexpected sub-expression')
            print(inStr)
            exit()
    elif len(sign_pi)>1:
        print('Error, unexpected sub-expression, found %d π'%len(sign_pi))
        print(inStr)
        exit()
    
    # check sqrt
    sign_sqrt=re.findall("[√]",inStr)
    if len(sign_sqrt)==1:
        str_list=list(inStr)
        first_sign=str_list.pop(0)
        if first_sign!='√':
            print('Error, unexpected sub-expression, √ is not at the leading site')
            print(inStr)
            exit()
        newExp=''.join(str_list)
        return np.sqrt(evalMathExpression(newExp))
    elif len(sign_sqrt)>1:
        print('Error, unexpected sub-expression, found %d √'%len(sign_sqrt))
        print(inStr)
        exit()
    
    return int(inStr)

    
def readInDoublePointGroupInfo(filePath,SYMPREC=1e-5):
    with open(filePath,'r') as fin:
        data=[line for line in fin.read().split('\n') if line!='' and line!='\t']
        title_line=[x for x in data[0].split(' ') if x!='']
        print('file of %s'%data[0])

        # detect basic info. of character tables
        index0=2
        cnt=0
        irrep_notation_cnt=0
        while(True):
            detect_line_index=index0+cnt
            detectLine=data[detect_line_index]
            form=re.findall("[(,C]",detectLine)
            if len(form)==0:
                break
            cnt+=1
            if form[0]=='(':
                irrep_notation_cnt+=1
        nIrrep=cnt-irrep_notation_cnt
        print('num. of classes: %d'%nIrrep)
        print('num. of notation types: %d'%irrep_notation_cnt)

        # read in character tables
        irrep_notation_list=[[] for i in range(irrep_notation_cnt)] # irrep_notation_list[type of notation][index of irrep]
        character_table=[] # character_table[index of irrep][index of class]
        iline=2+cnt
        for i_irrep in range(nIrrep):
            for i_notation in range(irrep_notation_cnt):
                irrep_notation_list[i_notation].append(data[iline])
                iline+=1
            character_vec=[]
            for i_class in range(nIrrep):
                character_vec.append(evalMathExpression(data[iline]))
                iline+=1
            character_table.append(character_vec)
        #print(character_table[-1])
        
        # read in class structures
        iline=iline+irrep_notation_cnt+1
        class_structure=[] # class_structure[class index][group ele index]
        operator_class_type_dic={}
        cnt=0
        while(True):
            line=data[iline].split(':')
            if len(line)!=2:
                break
            operatorList=[x for x in line[1].split(',') if x!='']
            class_structure.append(operatorList)
            for operater_str in operatorList:
                operator_class_type_dic[operater_str.strip()]=cnt
                #print(operater_str)
            #operator_class_type_list.extend([cnt]*len(operatorList))
            cnt+=1
            #print(operatorList)
            iline+=1
        #print(operator_class_type_dic['1'])
        #exit()
        groupOrder=sum([len(class_operator_list) for class_operator_list in class_structure])
        print("order of double group: %d"%groupOrder)
        #print(operator_class_type_list)

        # seek operator part
        while(True):
            if data[iline]=='N': break
            iline+=1
        iline+=(3+nIrrep)
        assert data[iline]=='1', 'suppose counter 1 after skip the irrep name, now get %s'%data[iline]
        #print(data[iline])

        # read in operators and irreps
        irrep=[]
        Seitz_symbol=[]
        operator_list=[]
        for ioperator in range(groupOrder):
            pg_operator=PointGroupOperator(ioperator,'',[],[],0,[],[])
            iline+=1 # skip operator index
            # read in space rotation matrix
            iline+=1 # skip (
            rotation=[]
            for irow in range(3):
                rotation.append([int(x) for x in data[iline].split('\t')])
                iline+=1
            iline+=1 # skip )
            pg_operator.rotation_r=np.array(rotation)

            # read in spin rotation matrix
            iline+=1 # skip (
            rotation_spin=[]
            for irow in range(2):
                rotation_spin.append([evalMathExpression(x) for x in data[iline].split('\t')])
                iline+=1
            iline+=1 # skip )
            pg_operator.rotation_s=np.array(rotation_spin,dtype=complex)
            
            # read in Seitz symbol
            Seitz_symbol.append(data[iline].strip())
            pg_operator.SeitzSymbol=data[iline].strip()
            iline+=1

            op_rep=[] # op_rep[index of group ele][index of irrep]
            for i_irrep in range(nIrrep):
                #print('irrep %d'%i_irrep)
                #if data[iline]
                rep_matrix=[]
                tuple_list=re.findall('[(]',data[iline])
                if len(tuple_list)==1: #find (
                    iline+=1 # skip (
                    dim_rep=len([x for x in data[iline].split('\t') if x!='' and x!='\t'])
                    for idim in range(dim_rep):
                        rep_matrix.append([evalMathExpression(x) for x in data[iline].split('\t') if x!='' and x!='\t'])
                        iline+=1
                    #print(rep_matrix)
                    #print('after rep matrix construction',data[iline])
                    tuple_list=re.findall('[)]',data[iline])
                    assert len(tuple_list)==1, 'tuple is not paired %s'%data[iline]
                    iline+=1 #skip )
                    #print('after skipping )',data[iline])
                    #op_rep.append(rep_matrix)
                elif len(tuple_list)==0:
                    rep_matrix.append([evalMathExpression(data[iline])])
                    iline+=1
                    #op_rep.append(rep_matrix)
                op_rep.append(np.array(rep_matrix))
            irrep.append(op_rep)
            #print(data[iline])
            #exit()
            pg_operator.irrepList=op_rep
            pg_operator.characterVec=np.array([sum(np.diag(rep)) for rep in op_rep])
            #print(np.diag(op_rep[0]))
            #exit()
            #print([np.diag(rep) for rep in op_rep])
            #print([sum(np.diag(rep)) for rep in op_rep])
            #exit()
            operator_list.append(pg_operator)
        print(data[iline])

        print('verification data...')
        # construct class list
        class_list=[]
        for ic in range(nIrrep):
            class_list.append(PointGroupClass(ic,[],np.array([character_table[i_irrep][ic] for i_irrep in range(nIrrep)])))
            #print([character_table[i_irrep][ic] for i_irrep in range(nIrrep)])
        for pg_operator in operator_list:
            corrClassIndex=operator_class_type_dic[pg_operator.SeitzSymbol]
            class_list[corrClassIndex].operatorList.append(pg_operator)
            
            operator_character_vec=pg_operator.characterVec
            class_character_vec=class_list[corrClassIndex].characterVec
            if sum(abs(operator_character_vec-class_character_vec))>SYMPREC:
                print("Error: group operator characters inconsist with class characters")
                print(operator_character_vec)
                print(class_character_vec)
                exit()
        print('success')
    return {"operators":operator_list, "class":class_list}

#readInDoublePointGroupInfo('./DPGREP/32.txt')
#value=evalMathExpression('(1-i√3)/2')
#print(value)#,-1j*np.sqrt(2)/2)