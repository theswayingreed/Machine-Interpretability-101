# Summary 
- Rent a machine in vast.ai. Cost $0.18/hr
- 


# References
## Vast Onstart Script

env >> /etc/environment
mkdir -p ${DATA_DIRECTORY:-/workspace}
cd /root || exit 1
[ -d Machine-Interpretability-101 ] || git clone https://github.com/theswayingreed/Machine-Interpretability-101.git
git -C Machine-Interpretability-101 pull --ff-only || true
bash Machine-Interpretability-101/setup/vast_setup.sh
