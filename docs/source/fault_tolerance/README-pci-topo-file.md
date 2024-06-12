# **Providing a PCI Topology File for GPU and NIC Topology Detection**

## **Overview**
In certain environments, such as virtual machines (VMs) provided by cloud service providers (CSPs), the PCI device tree may not be fully populated. When this occurs, traversing the system PCI device tree to determine the GPU and NIC topology is not viable.  

To work around this limitation, users can specify a pre-defined PCI topology file using the following option:  

```
–ft-param-pci-topo-file=<pci_topo_file>
```
where `<pci_topo_file>` is an XML file describing the PCI topology.

## **XML Format Requirements**
The PCI topology file follows a structured XML format with the following key elements:

1. **`<cpu>` Block:**  
   - Each CPU in the system is represented by a `<cpu>` block.
   
2. **`<pci>` Bridge Block:**  
   - Within each `<cpu>` block, there are one or more `<pci>` blocks that represent PCI bridges.  
   - Each PCI bridge has a unique `busid` attribute.
   
3. **GPU and IB PCI Devices:**  
   - Within each PCI bridge block, the `<pci>` elements represent GPU and InfiniBand (IB) devices.  
   - Each device has its own `busid` and attributes such as `class`, `link_speed`, and `link_width`.

### **Example 1: Single PCI Bridge per CPU**
In this example, each CPU has a single PCI bridge, connecting multiple GPUs and IB devices.

```xml
<cpu ...>
  <pci busid="{PCI-BRIDGE-BUSID}" ...>          
    <pci busid="{GPU0_BUSID}" .../>
    <pci busid="{IB0_BUSID}" .../>
    <pci busid="{GPU1_BUSID}" .../>
    <pci busid="{IB1_BUSID}" .../>
  </pci>
</cpu>
<cpu ...>
  <pci busid="{PCI-BRIDGE-BUSID}" ...>
    <pci busid="{GPU2_BUSID}" .../>
    <pci busid="{IB2_BUSID}" .../>
    <pci busid="{GPU3_BUSID}" .../>
    <pci busid="{IB3_BUSID}" .../>
  </pci>
</cpu>
```

### **Example 1: Example 2: Multiple PCI Bridges per CPU**
This example shows a topology where each CPU has multiple PCI bridges, with GPUs and IB devices distributed across them.

```xml
<cpu ...>
  <pci busid="{PCI-BRIDGE-BUSID}" ...> 
    <pci busid="{GPU0_BUSID}" .../>
    <pci busid="{IB0_BUSID}" .../>
  </pci>
  <pci busid="{PCI-BRIDGE-BUSID}" ...>
    <pci busid="{GPU1_BUSID}" .../>
    <pci busid="{IB1_BUSID}" .../>
  </pci>
</cpu>
<cpu ...>
  <pci busid="{PCI-BRIDGE-BUSID}" ...> 
    <pci busid="{GPU2_BUSID}" .../>
    <pci busid="{IB2_BUSID}" .../>
  </pci>
  <pci busid="{PCI-BRIDGE-BUSID}" ...>
    <pci busid="{GPU3_BUSID}" .../>
    <pci busid="{IB3_BUSID}" .../>
  </pci>
</cpu>
```

## **Reference Example**
For a detailed working example, refer to the [NDv4 topology file](https://github.com/Azure/azhpc-images/blob/master/topology/ndv4-topo.xml) in the Azure HPC images repository.

