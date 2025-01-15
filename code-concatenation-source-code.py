# Code for PQC project by Mahtab

# import the necessary tools
import cirq.circuits
from qualtran import Bloq, CompositeBloq, BloqBuilder, Signature, Register, QBit, QAny
# from qualtran.drawing import show_bloq
from qualtran.bloqs.basic_gates import CNOT, Hadamard, XGate, Toffoli, ZGate 
from qualtran.bloqs.mcmt import MultiTargetCNOT, MultiControlPauli, multi_control_multi_target_pauli
from typing import *
from qualtran.drawing import get_musical_score_data, draw_musical_score
from qualtran import SoquetT
import cirq
import numpy as np
import matplotlib.pyplot as plt
import sympy
import attrs
from qualtran.cirq_interop import BloqAsCirqGate, cirq_optree_to_cbloq
from cirq.contrib.svg import SVGCircuit

# Patterns for Syndrome and Recovery
CVS = [
(1, 0, 0, 0, 0, 0),  #X1
(1, 1, 0, 0, 0, 0),  #X2
(0, 1, 0, 0, 0, 0),  #X3
(0, 0, 1, 0, 0, 0),  #X4
(0, 0, 1, 1, 0, 0),  #X5
(0, 0, 0, 1, 0, 0),  #X6
(0, 0, 0, 0, 1, 0),  #X7
(0, 0, 0, 0, 1, 1),  #X8
(0, 0, 0, 0, 0, 1),  #X9
(1, 0),  #Z1, Z2, Z3
(1, 1),  #Z6, Z5, Z4
(0, 1),  #Z7, Z8, Z9
]


# Bloq for (unconcatenated) Shor code
# In this code "logical" is the Soquet with the 9 qubits, and "qubits" is the form of "logical" that's modified during the code. Same goes for "ancilla" and "a".
@attrs.frozen
class ShorCodeAll(Bloq):
    @property
    def signature(self):
        return Signature.build(logical=9, ancilla=8)

    
    def build_composite_bloq(self, bb: BloqBuilder, *, logical: SoquetT, ancilla: SoquetT) -> Dict[str, SoquetT]: 

        # Initialize the data qubit to |+> state
        qubits = bb.split(logical)
        qubits[0] = bb.add(Hadamard(), q=qubits[0])
        qubits = bb.join(qubits)

        # Encoding 
        qubits= bb.add_from(ShorEncode(), logical=qubits)[0]

        # syndrome measurements
        qubits, a = bb.add_from(ShorSyndrome(), logical=qubits, ancilla=ancilla)

        # recovery 
        qubits, a = bb.add_from(ShorRecovery(), logical=qubits, ancilla=a)

        # decoding step
        qubits= bb.add_from(ShorDecode(), logical=qubits)[0]

        # return the error corrected qubits
        return {'logical': qubits, 'ancilla': a}
    
    
@attrs.frozen    
class ShorEncode(Bloq):
    @property
    def signature(self):
        return Signature.build(logical=9)

    
    def build_composite_bloq(self, bb: BloqBuilder, *, logical: SoquetT) -> Dict[str, SoquetT]: 

        qubits = bb.split(logical)

        # Entangle qubits 0, 3, and 6 for phase flip error detection
        qubits[0], qubits[3] = bb.add(CNOT(), ctrl = qubits[0], target=qubits[3])
        qubits[0], qubits[6] = bb.add(CNOT(), ctrl = qubits[0], target=qubits[6])
        
        # the three groups of three bit flip detecting qubits
        for i in [0, 3, 6]:
            # add hadamard to the qubiti
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
            # CNOT between qubiti and qubiti+1
            qubits[i], qubits[i+1] = bb.add(CNOT(), ctrl=qubits[i], target=qubits[i+1])
            # CNOT between qubiti and qubiti+2
            qubits[i], qubits[i+2] = bb.add(CNOT(), ctrl=qubits[i], target=qubits[i+2])


        return {'logical': bb.join(qubits)}
    

@attrs.frozen
class ShorSyndrome(Bloq):
    @property
    def signature(self):
        return Signature.build(logical=9, ancilla=8)

    
    def build_composite_bloq(self, bb: BloqBuilder, *, logical: SoquetT, ancilla:SoquetT) -> Dict[str, SoquetT]: 
        
        qubits = bb.split(logical)
        a = bb.split(ancilla)

        for i in range(8):
            # apply Hadamards on all the ancillas
            a[i] = bb.add(Hadamard(), q=a[i])
        
        

        multiCNOT6 = MultiTargetCNOT(bitsize=6)

        # syndrome X3X4X5X6X7X8 with ancilla7
        a[7], trgt = bb.add(multiCNOT6, control=a[7], targets=bb.join(qubits[3:9]))

        qubits[3:9] = bb.split(trgt)
         
        
        # syndrome X0X1X2X3X4X5 with the ancilla6
        a[6], trgt = bb.add(multiCNOT6, control=a[6], targets=bb.join(qubits[0:6]))

        qubits[0:6] = bb.split(trgt)


        # syndrome Z7Z8 with ancilla5 (applied as Hadamard sandwiched multi-target CNOT)
        for i in [7, 8]:
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        
        multiCNOT2 = MultiTargetCNOT(bitsize=2)
        a[5], trgt = bb.add(multiCNOT2, control=a[5], targets=bb.join(qubits[7:9]))
        qubits[7:9] = bb.split(trgt)

        for i in [7, 8]:
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        

        # syndrome Z6Z7 with ancilla4
        for i in [6, 7]:
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        
        a[4], trgt = bb.add(multiCNOT2, control=a[4], targets=bb.join(qubits[6:8]))
        qubits[6:8] = bb.split(trgt)

        for i in [6, 7]:
            qubits[i] = bb.add(Hadamard(), q=qubits[i])


        # syndrome Z4Z5 with ancilla3
        for i in [4, 5]:
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        
        a[3], trgt = bb.add(multiCNOT2, control=a[3], targets=bb.join(qubits[4:6]))
        qubits[4:6] = bb.split(trgt)

        for i in [4, 5]:
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        

        # syndrome Z3Z4 with ancilla2
        for i in [3, 4]:
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        
        a[2], trgt = bb.add(multiCNOT2, control=a[2], targets=bb.join(qubits[3:5]))
        qubits[3:5] = bb.split(trgt)

        for i in [3, 4]:
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        
        
        # syndrome Z1Z2 with ancilla1
        for i in [1, 2]:
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        
        a[1], trgt = bb.add(multiCNOT2, control=a[1], targets=bb.join(qubits[1:3]))
        qubits[1:3] = bb.split(trgt)

        for i in [1, 2]:
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        
        
        # syndrome Z0Z1 with ancilla0
        for i in [0, 1]:
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        
        a[0], trgt = bb.add(multiCNOT2, control=a[0], targets=bb.join(qubits[0:2]))
        qubits[0:2] = bb.split(trgt)

        for i in [0, 1]:
            qubits[i] = bb.add(Hadamard(), q=qubits[i])
        

        for i in range(8):
            # apply Hadamards on all the ancillas
            a[i] = bb.add(Hadamard(), q=a[i])
        
        return {'logical': bb.join(qubits), 'ancilla': bb.join(a)}




@attrs.frozen
class ShorRecovery(Bloq):
    @property
    def signature(self):
        return Signature.build(logical=9, ancilla=8)
 
    def build_composite_bloq(self, bb: BloqBuilder, logical: SoquetT, ancilla: SoquetT):

        qubits = bb.split(logical)
        a = bb.split(ancilla)

        # correct for X errors
        for i in range(9):
            superCNOT = MultiControlPauli(cvs=CVS[i], target_gate=cirq.X)
            a[0:6], qubits[i] = bb.add(superCNOT, controls=a[0:6], target=qubits[i])
        
        # correct for Z errors
        for (i, j) in [(9, 0), (10, 3), (11, 6)]:
            superCZ = MultiControlPauli(cvs=CVS[i], target_gate=cirq.Z)
            a[6:8], qubits[j] = bb.add(superCZ, controls=a[6:8], target=qubits[j])
        
        return {'logical': bb.join(qubits), 'ancilla': bb.join(a)}
    


@attrs.frozen
class ShorDecode(Bloq):
    @property
    def signature(self):
        return Signature.build(logical=9)
 
    def build_composite_bloq(self, bb: BloqBuilder, logical: SoquetT):

        qubits = bb.split(logical)

        for i in [0, 3, 6]:
            # apply CNOTs
            qubits[i], qubits[i+1] = bb.add(CNOT(), ctrl=qubits[i], target=qubits[i+1])
            qubits[i], qubits[i+2] = bb.add(CNOT(), ctrl=qubits[i], target=qubits[i+2])

            # apply Hadamards
            qubits[i] = bb.add(Hadamard(), q=qubits[i])


        # CNOT between q0&q3 and q0&q6
        qubits[0], qubits[3] = bb.add(CNOT(), ctrl = qubits[0], target=qubits[3])
        qubits[0], qubits[6] = bb.add(CNOT(), ctrl = qubits[0], target=qubits[6])


        return {'logical': bb.join(qubits)}
    


@attrs.frozen
class logicalX(Bloq):
    @property
    def signature(self):
        return Signature.build(logical=9)
 
    def build_composite_bloq(self, bb: BloqBuilder, logical: SoquetT):

        qubits = bb.split(logical)
        
        # logical X = XXXXXXXXX
        for i in range(9):
            qubits[i] = bb.add(XGate(), q=qubits[i])

        return {'logical': bb.join(qubits)}



@attrs.frozen
class logicalZ(Bloq):
    @property
    def signature(self):
        return Signature.build(logical=9)
 
    def build_composite_bloq(self, bb: BloqBuilder, logical: SoquetT):
        
        qubits = bb.split(logical)
        
        # logical Z = ZZZZZZZZZ
        for i in range(9):
            qubits[i] = bb.add(ZGate(), q=qubits[i])

        return {'logical': bb.join(qubits)}



@attrs.frozen
class logicalH(Bloq):
    n: int
    @property
    def signature(self):
        return Signature.build(logical=self.n)
 
    def build_composite_bloq(self, bb: BloqBuilder, logical: SoquetT):
        
        qubits = bb.split(logical)
        
        # logical H = HHHHHHHHH
        for i in range(self.n):
            qubits[i] = bb.add(Hadamard(), q=qubits[i])

        logical = bb.join(qubits)

        return {'logical': logical}



@attrs.frozen
class logicalCNOT(Bloq):
    @property
    def signature(self):
        return Signature.build(lctrl=9, ltarget=9)
 
    def build_composite_bloq(self, bb: BloqBuilder, lctrl: SoquetT, ltarget: SoquetT):
        
        c = bb.split(lctrl)
        t = bb.split(ltarget)

        for i in range(9):
            c[i], t[i] = bb.add(CNOT(), ctrl=c[i], target=t[i])
        
        return {'lctrl': bb.join(c), 'ltarget': bb.join(t)}

@attrs.frozen
class logical_6TargetCNOT(Bloq):
    @property
    def signature(self):
        return Signature.build(lctrl=1, ltarget1=9, ltarget2=9, ltarget3=9, ltarget4=9, ltarget5=9, ltarget6=9)
 
    def build_composite_bloq(self, bb: BloqBuilder, lctrl: SoquetT, ltarget1: SoquetT, ltarget2: SoquetT, ltarget3: SoquetT, ltarget4: SoquetT, ltarget5: SoquetT, ltarget6: SoquetT):

        ltargets = [ltarget1, ltarget2, ltarget3, ltarget4, ltarget5, ltarget6]
        multiCNOT9 = MultiTargetCNOT(bitsize=9)
        c = lctrl
        t = []


        for i, target in enumerate(ltargets):
            c, tr = bb.add(multiCNOT9, control=c, targets=target)
            t.append(tr)
        
        t = np.array(t)
        
        return {'lctrl': c, 'ltarget1': t[0], 'ltarget2': t[1], 'ltarget3': t[2], 'ltarget4': t[3], 'ltarget5': t[4], 'ltarget6': t[5]}
 


@attrs.frozen
class logical_2TargetCNOT(Bloq):
    @property
    def signature(self):
        return Signature.build(lctrl=1, ltarget1=9, ltarget2=9)
 
    def build_composite_bloq(self, bb: BloqBuilder, lctrl: SoquetT, ltarget1: SoquetT, ltarget2: SoquetT):

        ltargets = [ltarget1, ltarget2]
        multiCNOT9 = MultiTargetCNOT(bitsize=9)
        c = lctrl
        t = []


        for i, target in enumerate(ltargets):
            c, tr = bb.add(multiCNOT9, control=c, targets=target)
            t.append(tr)
        
        t = np.array(t)
        
        return {'lctrl': c, 'ltarget1': t[0], 'ltarget2': t[1]}
 


@attrs.frozen
class logical_6controlToffoli(Bloq):
    cvs: tuple 
    @property
    def signature(self):
        return Signature.build(lctrl1=1, lctrl2=1, lctrl3=1, lctrl4=1, lctrl5=1, lctrl6=1, ltarget=9)
 
    def build_composite_bloq(self, bb: BloqBuilder, lctrl1: SoquetT, lctrl2: SoquetT, lctrl3: SoquetT, lctrl4: SoquetT, lctrl5: SoquetT, lctrl6: SoquetT, ltarget: SoquetT):

        ctrls = [lctrl1, lctrl2, lctrl3, lctrl4, lctrl5, lctrl6]
        target = bb.split(ltarget)
        superCNOT = MultiControlPauli(cvs=self.cvs, target_gate=cirq.X)

        for i in range(9):
            ctrls, target[i] = bb.add(superCNOT, controls=ctrls, target=target[i])
        

        return {'lctrl1': ctrls[0], 'lctrl2': ctrls[1], 'lctrl3': ctrls[2], 'lctrl4': ctrls[3], 'lctrl5': ctrls[4], 'lctrl6': ctrls[5], 'ltarget': bb.join(target)}
            

@attrs.frozen
class logical_2controlCZ(Bloq):
    cvs: tuple 
    @property
    def signature(self):
        return Signature.build(lctrl1=1, lctrl2=1, ltarget=9)
 
    def build_composite_bloq(self, bb: BloqBuilder, lctrl1: SoquetT, lctrl2: SoquetT, ltarget: SoquetT):

        ctrls = [lctrl1, lctrl2]
        target = bb.split(ltarget)
        superCZ = MultiControlPauli(cvs=self.cvs, target_gate=cirq.Z)

        for i in range(9):
            ctrls, target[i] = bb.add(superCZ, controls=ctrls, target=target[i])
        

        return {'lctrl1': ctrls[0], 'lctrl2': ctrls[1], 'ltarget': bb.join(target)}
            


# concatenated Shor code bloqs.
@attrs.frozen
class concatenatedShorAll(Bloq):
    @property
    def signature(self):
        return Signature.build(logicals=81, ancillas=80)

 
    def build_composite_bloq(self, bb: BloqBuilder, logicals: SoquetT, ancillas: SoquetT):

        # Encoding
        l, a = bb.add_from(concatenatedShor_encode(), logicals=logicals, ancillas=ancillas)

        # syndrome measurements
        l, a = bb.add_from(concatenatedShor_syndrome(), logicals=l, ancillas=a)

        # recovery
        l, a = bb.add_from(concatenatedShor_recovery(), logicals=l, ancillas=a)

        # decoding
        l, a = bb.add_from(concatenatedShor_decode(), logicals=l, ancillas=a)

        return {'logicals': l, 'ancillas': a}




@attrs.frozen
class concatenatedShor_encode(Bloq):
    @property
    def signature(self):
        return Signature.build(logicals=81, ancillas=80)

 
    def build_composite_bloq(self, bb: BloqBuilder, logicals: SoquetT, ancillas: SoquetT):
        
        # split the register to 81 qubits
        l = bb.split(logicals)

        # join the corresponding qubits to make the 9 logical qubits
        logs = [] # array to store the logical qubits
        for i in range(9):
            start = i * 9
            logs.append(bb.join(l[start:start+9]))
        
        logs = np.array(logs)

        # Do normal Shor encoding on each of the logical qubits and ancillas 
        for i in range(9):
            logs[i] = bb.add_from(ShorEncode(), logical=logs[i])[0]
        
        # Entangle logicals 0, 3, and 6 
        logs[0], logs[3] = bb.add_from(logicalCNOT(), lctrl=logs[0], ltarget=logs[3])
        logs[0], logs[6] = bb.add_from(logicalCNOT(), lctrl=logs[0], ltarget=logs[6])


        # Entangle 0->1, 2; 3->4, 5; 6->7, 8
        for i in [0, 3, 6]:
            # add_from gives a tuple, so when you have one-qubit gate, you need to have [0]
            logs[i] = bb.add_from(logicalH(9), logical=logs[i])[0]
            logs[i], logs[i+1] = bb.add_from(logicalCNOT(), lctrl=logs[i], ltarget=logs[i+1])
            logs[i], logs[i+2] = bb.add_from(logicalCNOT(), lctrl=logs[i], ltarget=logs[i+2])
        
        
        # split the logical qubits back into l
        l = []
        for i in range(9):
            l.append(bb.split(logs[i]))
        
        # flatten the array (to get a 1d array)
        l = np.array([item for sublist in l for item in sublist])
        
        return {'logicals': bb.join(l), 'ancillas': ancillas}
    



@attrs.frozen
class concatenatedShor_syndrome(Bloq):
    @property
    def signature(self):
        return Signature.build(logicals=81, ancillas=80)

 
    def build_composite_bloq(self, bb: BloqBuilder, logicals: SoquetT, ancillas: SoquetT):
        
        # split the register to 81 qubits
        l = bb.split(logicals)

        # join the corresponding qubits to make the 9 logical qubits
        logs = [] # array to store the logical qubits
        for i in range(9):
            start = i * 9
            logs.append(bb.join(l[start:start+9]))
        
        logs = np.array(logs)

        # split the ancillas register to 80 qubits
        a = bb.split(ancillas)

        # join the corresponding qubits to make 10 soquets of 8 ancillas each
        ancis = [] # array to store the "logical" ancillas
        for i in range(10):
            start = i * 8
            ancis.append(bb.join(a[start:start+8]))
        
        ancis = np.array(ancis)


        # do normal syndrome measurement on each logical qubit
        for i in range(9):
            logs[i], ancis[i] = bb.add_from(ShorSyndrome(), logical=logs[i], ancilla=ancis[i])
        
        # Hadmards on the last logical ancilla
        ancis[9] = bb.add_from(logicalH(8), logical=ancis[9])[0]
        last_ancilla = bb.split(ancis[9])   # the split version of soquet representing the last 8 ancillas

        # syndrome X3X4X5X6X7X8 with ancilla79 and syndrome X0X1X2X3X4X5 with ancilla78
        last_ancilla[7], logs[3], logs[4], logs[5], logs[6], logs[7], logs[8] = bb.add_from(logical_6TargetCNOT(), lctrl=last_ancilla[7], ltarget1=logs[3], ltarget2=logs[4], ltarget3=logs[5], ltarget4=logs[6], ltarget5=logs[7], ltarget6=logs[8])

        last_ancilla[6], logs[0], logs[1], logs[2], logs[3], logs[4], logs[5] = bb.add_from(logical_6TargetCNOT(), lctrl=last_ancilla[6], ltarget1=logs[0], ltarget2=logs[1], ltarget3=logs[2], ltarget4=logs[3], ltarget5=logs[4], ltarget6=logs[5])

        # perform the ZZ syndromes
        # Z0Z1: last_ancilla0, Z1Z2: last_ancilla1
        for i in range(2):
            # Hadamards
            logs[i] = bb.add_from(logicalH(9), logical=logs[i])[0]
            logs[i+1] = bb.add_from(logicalH(9), logical=logs[i+1])[0]

            # 2target CNOTS
            last_ancilla[i], logs[i], logs[i+1] = bb.add_from(logical_2TargetCNOT(), lctrl=last_ancilla[i], ltarget1=logs[i], ltarget2=logs[i+1])

            # Hadamards
            logs[i] = bb.add_from(logicalH(9), logical=logs[i])[0]
            logs[i+1] = bb.add_from(logicalH(9), logical=logs[i+1])[0]
        
        # Z3Z4: last_ancilla2, Z4Z5: last_ancilla3
        for i in range(2, 4):
            # Hadamards
            logs[i+1] = bb.add_from(logicalH(9), logical=logs[i+1])[0]
            logs[i+2] = bb.add_from(logicalH(9), logical=logs[i+2])[0]

            # 2target CNOTS
            last_ancilla[i], logs[i+1], logs[i+2] = bb.add_from(logical_2TargetCNOT(), lctrl=last_ancilla[i], ltarget1=logs[i+1], ltarget2=logs[i+2])

            # Hadamards
            logs[i+1] = bb.add_from(logicalH(9), logical=logs[i+1])[0]
            logs[i+2] = bb.add_from(logicalH(9), logical=logs[i+2])[0]

        # Z6Z7: last_ancilla4, Z7Z8: last_ancilla5
        for i in range(4, 6):
            # Hadamards
            logs[i+2] = bb.add_from(logicalH(9), logical=logs[i+2])[0]
            logs[i+3] = bb.add_from(logicalH(9), logical=logs[i+3])[0]

            # 2target CNOTS
            last_ancilla[i], logs[i+2], logs[i+3] = bb.add_from(logical_2TargetCNOT(), lctrl=last_ancilla[i], ltarget1=logs[i+2], ltarget2=logs[i+3])

            # Hadamards
            logs[i+2] = bb.add_from(logicalH(9), logical=logs[i+2])[0]
            logs[i+3] = bb.add_from(logicalH(9), logical=logs[i+3])[0]
        
        
        # join back the last ancilla
        ancis[9] = bb.join(last_ancilla)   
        # Hadmards on the last logical ancilla
        ancis[9] = bb.add_from(logicalH(8), logical=ancis[9])[0]


        # split the logical qubits back into l
        l = []
        for i in range(9):
            l.append(bb.split(logs[i]))
        
        # flatten the array (to get a 1d array)
        l = np.array([item for sublist in l for item in sublist])

        # split the ancilla qubits back into a
        a = []
        for i in range(10):
            a.append(bb.split(ancis[i]))
        
        # flatten the array (to get a 1d array)
        a = np.array([item for sublist in a for item in sublist])
        
        return {'logicals': bb.join(l), 'ancillas': bb.join(a)}



@attrs.frozen
class concatenatedShor_recovery(Bloq):
    @property
    def signature(self):
        return Signature.build(logicals=81, ancillas=80)

 
    def build_composite_bloq(self, bb: BloqBuilder, logicals: SoquetT, ancillas: SoquetT):
        
        # split the register to 81 qubits
        l = bb.split(logicals)

        # join the corresponding qubits to make the 9 logical qubits
        logs = [] # array to store the logical qubits
        for i in range(9):
            start = i * 9
            logs.append(bb.join(l[start:start+9]))
        
        logs = np.array(logs)

        # split the ancillas register to 80 qubits
        a = bb.split(ancillas)

        # join the corresponding qubits to make 10 soquets of 8 ancillas each
        ancis = [] # array to store the "logical" ancillas
        for i in range(10):
            start = i * 8
            ancis.append(bb.join(a[start:start+8]))
        
        ancis = np.array(ancis)

        # do recovery on the physical qubits of each logical qubit
        for i in range(9):
            logs[i], ancis[i] = bb.add_from(ShorRecovery(), logical=logs[i], ancilla=ancis[i])
        

        last_ancilla = bb.split(ancis[9])   # the split version of soquet representing the last 8 ancillas
        # correct for X errors of the 9 logical qubits
        for i in range(9):
            last_ancilla[0], last_ancilla[1], last_ancilla[2], last_ancilla[3], last_ancilla[4], last_ancilla[5], logs[i] = bb.add_from(logical_6controlToffoli(CVS[i]), lctrl1=last_ancilla[0], lctrl2=last_ancilla[1], lctrl3=last_ancilla[2], lctrl4=last_ancilla[3], lctrl5=last_ancilla[4], lctrl6=last_ancilla[5], ltarget=logs[i])


        # correct for Z errors of the 9 logical qubits
        for (i, j) in [(9, 0), (10, 3), (11, 6)]:
            last_ancilla[6], last_ancilla[7], logs[j] = bb.add_from(logical_2controlCZ(CVS[i]), lctrl1=last_ancilla[6], lctrl2=last_ancilla[7], ltarget=logs[j])
        

         # join back the last ancilla
        ancis[9] = bb.join(last_ancilla)   


        # split the logical qubits back into l
        l = []
        for i in range(9):
            l.append(bb.split(logs[i]))
        
        # flatten the array (to get a 1d array)
        l = np.array([item for sublist in l for item in sublist])

        # split the ancilla qubits back into a
        a = []
        for i in range(10):
            a.append(bb.split(ancis[i]))
        
        # flatten the array (to get a 1d array)
        a = np.array([item for sublist in a for item in sublist])
        
        return {'logicals': bb.join(l), 'ancillas': bb.join(a)}    



@attrs.frozen
class concatenatedShor_decode(Bloq):
    @property
    def signature(self):
        return Signature.build(logicals=81, ancillas=80)

 
    def build_composite_bloq(self, bb: BloqBuilder, logicals: SoquetT, ancillas: SoquetT):
        
        # split the register to 81 qubits
        l = bb.split(logicals)

        # join the corresponding qubits to make the 9 logical qubits
        logs = [] # array to store the logical qubits
        for i in range(9):
            start = i * 9
            logs.append(bb.join(l[start:start+9]))
        
        logs = np.array(logs)

        # Do normal Shor decoding on each of the logical qubits and ancillas 
        for i in range(9):
            logs[i] = bb.add_from(ShorDecode(), logical=logs[i])[0]
        

        for i in [0, 3, 6]:
            logs[i], logs[i+1] = bb.add_from(logicalCNOT(), lctrl=logs[i], ltarget=logs[i+1])
            logs[i], logs[i+2] = bb.add_from(logicalCNOT(), lctrl=logs[i], ltarget=logs[i+2])
            # add_from gives a tuple, so when you have one-qubit gate, you need to have [0]
            logs[i] = bb.add_from(logicalH(9), logical=logs[i])[0]

        
        logs[0], logs[3] = bb.add_from(logicalCNOT(), lctrl=logs[0], ltarget=logs[3])
        logs[0], logs[6] = bb.add_from(logicalCNOT(), lctrl=logs[0], ltarget=logs[6])
        
        
        # split the logical qubits back into l
        l = []
        for i in range(9):
            l.append(bb.split(logs[i]))
        
        # flatten the array (to get a 1d array)
        l = np.array([item for sublist in l for item in sublist])
        
        return {'logicals': bb.join(l), 'ancillas': ancillas}



# Different visualizations of the circuits:

# show the concatenated Shor decode circuit using Qualtran
shorbloq = concatenatedShor_decode().decompose_bloq()
msd = get_musical_score_data(shorbloq)
fig, ax = draw_musical_score(msd)
fig.set_figwidth(9)
plt.show() 

# show the not concatenated Shor code using SVG Cirq (SVG should be seen in jupyter notebook, it doesn't show in VS Code)
circuit, _ = ShorCodeAll().as_composite_bloq().to_cirq_circuit(
    logical=cirq.LineQubit.range(9), ancilla=cirq.LineQubit.range(9, 17)
)
op = next(circuit.all_operations())
shor_decomp_circuit = cirq.Circuit(cirq.decompose_once(op))
SVGCircuit(shor_decomp_circuit)



# show concatenated Shor code SVG Cirq
circuit, _ = concatenatedShorAll().as_composite_bloq().to_cirq_circuit(
    logicals=cirq.LineQubit.range(81), ancillas=cirq.LineQubit.range(81, 161)
)
op = next(circuit.all_operations())
shor_decomp_circuit = cirq.Circuit(cirq.decompose_once(op))
SVGCircuit(shor_decomp_circuit)


# show the concatenated Shor code circuit using Cirq print circuit
circuit, _ = concatenatedShorAll().as_composite_bloq().to_cirq_circuit(
    logicals=cirq.LineQubit.range(81), ancillas=cirq.LineQubit.range(81, 161)
)
op = next(circuit.all_operations())
shor_decomp_circuit = cirq.Circuit(cirq.decompose_once(op))
print(shor_decomp_circuit)



# Simulations for calculating the logical error rates

# Shor code bloq with a specific errro rate
@attrs.frozen
class ShorCodeAll_withError(Bloq):
    x: float
    @property
    def signature(self):
        return Signature.build(logical=9, ancilla=8)

    
    def build_composite_bloq(self, bb: BloqBuilder, *, logical: SoquetT, ancilla: SoquetT) -> Dict[str, SoquetT]: 

        # Initialize the data qubit to |+> state
        qubits = bb.split(logical)
        qubits[0] = bb.add(Hadamard(), q=qubits[0])
        qubits = bb.join(qubits)

        # Encoding 
        qubits= bb.add_from(ShorEncode(), logical=qubits)[0]

        # cirq circuit for introducing errors
        circuit = cirq.Circuit()
        all_qubits = cirq.LineQubit.range(9)
        for qubit in all_qubits:
            circuit.append([cirq.I(qubit)])
        
        # add noise model to the circuit 
        noisy = circuit.with_noise(cirq.depolarize(p=self.x))

        # turn curcuit into bloq
        noisyBloq = CompositeBloq.from_cirq_circuit(noisy)

        # add the noisyBloq to the circuit
        qubits = bb.split(qubits)
        qubits = bb.add_from(noisyBloq, qubits=qubits)[0]
        qubits = bb.join(qubits)

        # syndrome measurements
        qubits, a = bb.add_from(ShorSyndrome(), logical=qubits, ancilla=ancilla)

        # recovery 
        qubits, a = bb.add_from(ShorRecovery(), logical=qubits, ancilla=a)

        # decoding step
        qubits= bb.add_from(ShorDecode(), logical=qubits)[0]

        # return the error corrected qubits
        return {'logical': qubits, 'ancilla': a}
    


errorless_circuit, _ = ShorCodeAll().as_composite_bloq().to_cirq_circuit(
    logical=cirq.LineQubit.range(9), ancilla=cirq.LineQubit.range(9, 17)
)

# simulate the unconcatenated circuit without any errors 
# initialize simulator
simulator = cirq.Simulator()
# simulate the errorless curcuit
result = simulator.simulate(errorless_circuit)
errorless_state_vector = np.around(result.final_state_vector, 5)
print('state vector of Shor code without error:', errorless_state_vector, '\n')


# simulate the crcuit with different physical errors using the depolarize noise channel
physical_erros = np.linspace(0.01, 0.05, 50)

# numebr of runs
n = 100

logical_error_rates = []

# simulating circuit with errors only between encoding and syndrome
for i in physical_erros:
    logical_error_counts = 0    # number of times the error is not corrected
    # circuit with physical error of i
    error_circuit, _ = ShorCodeAll_withError(i).as_composite_bloq().to_cirq_circuit(
    logical=cirq.LineQubit.range(9), ancilla=cirq.LineQubit.range(9, 17))

    # run the simulation n times
    for _ in range(n):
        error_result = simulator.simulate(error_circuit)
        error_vector = np.around(error_result.final_state_vector, 5)
        #print(f'noisy vector {i}: {error_vector} \n')

        # check if the error was corrected 
        if not np.array_equal(error_vector, errorless_state_vector):
            logical_error_counts +=1

    # add the logical error rate for this i to the array
    rate = logical_error_counts / n
    logical_error_rates.append(rate)

    print(f'logical error rate for physical error {i}: {rate} \n')
    
logical_error_rates = np.array(logical_error_rates)

# plot the results
plt.figure(figsize=(10, 6))
plt.plot(physical_erros, logical_error_rates, marker='o', linestyle='-', color='b', label='Data points')
plt.title('Logical Error Rates vs Physical Error Probability: Error Only After Encoding and Before Syndrome')
plt.xlabel('Physical Error Probability')
plt.ylabel('Logical Error Rates')
plt.legend()
plt.grid(True)
plt.show()



# logical error rates when the errors are everywhere
error2_circuit, _ = ShorCodeAll().as_composite_bloq().to_cirq_circuit(
    logical=cirq.LineQubit.range(9), ancilla=cirq.LineQubit.range(9, 17)
)

error2_rates = []

for i in physical_erros:
    logical_error_counts = 0    # number of times the error is not corrected
    # circuit with physical error of i
    noisy = error2_circuit.unfreeze().with_noise(cirq.depolarize(p=i))

    # run the simulation n times
    for _ in range(n):
        error_result = simulator.simulate(noisy)
        error_vector = np.around(error_result.final_state_vector, 5)

        # check if the error was corrected 
        if not np.array_equal(error_vector, errorless_state_vector):
            logical_error_counts +=1

    # add the logical error rate for this i to the array
    rate = logical_error_counts / n
    error2_rates.append(rate)

    print(f'logical error rate for physical error {i}: {rate} \n')

error2_rates = np.array(error2_rates)

# plot the results
plt.figure(figsize=(10, 6))
plt.plot(physical_erros, error2_rates, marker='o', linestyle='-', color='b', label='Data points')
plt.title('Logical Error Rates vs Physical Error Probability: Error Everywhere')
plt.xlabel('Physical Error Probability')
plt.ylabel('Logical Error Rates')
plt.legend()
plt.grid(True)
plt.show()


