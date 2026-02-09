# BRDF Module Index & Quick Navigation

## 📌 Quick Links

### For Quick Review (30 minutes)
1. **Start Here**: [BRDF_INTEGRATION.md](BRDF_INTEGRATION.md) - Overview of what's new
2. **Run Examples**: `python examples/brdf_correction_example.py` 
3. **Full Guide**: [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md) - Complete documentation

### For Deep Dive (2 hours)
1. Read [BRDF_CALL_TRACE.md](BRDF_CALL_TRACE.md) - Understand the architecture
2. Study [src/brdf_correction.py](src/brdf_correction.py) - Read the code
3. Review [BRDF_DELIVERY_REPORT.md](BRDF_DELIVERY_REPORT.md) - Full specifications

### For Integration (1 hour)
1. Check [BRDF_INTEGRATION.md](BRDF_INTEGRATION.md) - How to use
2. Look at [examples/brdf_correction_example.py](examples/brdf_correction_example.py) - See patterns
3. Test with [data/sample/sample_tropomi_brdf.csv](data/sample/sample_tropomi_brdf.csv) - Real data

---

## 📁 File Structure

```
GitHub_Repo/
│
├── 📄 BRDF_DELIVERY_REPORT.md          ← Final delivery report (spec sheet)
├── 📄 BRDF_INTEGRATION.md              ← Integration guide (start here!)
├── 📄 BRDF_CALL_TRACE.md               ← Architecture & call hierarchy
│
├── 📁 src/
│   └── 🐍 brdf_correction.py           ← Core module (533 lines)
│       ├── ross_thick_kernel()
│       ├── li_sparse_kernel()
│       ├── brdf_correction()
│       ├── apply_multi_angle_correction()
│       ├── validate_brdf_inputs()
│       └── BRDF_degree_vectorized()
│
├── 📁 docs/
│   └── 📄 BRDF_GUIDE.md                ← Full documentation (430 lines)
│       ├── Theory & mathematics
│       ├── Installation guide
│       ├── Usage examples
│       ├── Data formats
│       ├── Performance tips
│       └── Troubleshooting
│
├── 📁 examples/
│   └── 🐍 brdf_correction_example.py   ← 4 working examples
│       ├── Example 1: Basic correction
│       ├── Example 2: Kernel inspection
│       ├── Example 3: Multi-angle correction
│       └── Example 4: Input validation
│
└── 📁 data/sample/
    └── 📊 sample_tropomi_brdf.csv      ← Test data (31 rows)
```

---

## 🎯 What Each File Does

### Core Module
**File**: `src/brdf_correction.py` (533 lines)

Implements BRDF corrections for TROPOMI SIF data:
- **ross_thick_kernel()**: Volumetric scattering (canopy effects)
- **li_sparse_kernel()**: Geometric scattering (shadow effects)
- **brdf_correction()**: Main correction function
- **apply_multi_angle_correction()**: Generate multi-angle SIF/VI

### Documentation Files

**1. BRDF_INTEGRATION.md** (280 lines)
- What's new in this module
- Quick integration examples
- Comparison with original code
- Migration guide

**2. docs/BRDF_GUIDE.md** (430 lines)
- Complete theory and equations
- Installation instructions
- Detailed usage guide
- Data format specifications
- Performance optimization
- Troubleshooting solutions

**3. BRDF_CALL_TRACE.md** (460 lines)
- Function call hierarchy
- Integration with other modules
- Performance characteristics
- Memory usage analysis
- Debug tracing examples

**4. BRDF_DELIVERY_REPORT.md** (400 lines)
- Detailed specifications
- File breakdown
- Testing checklist
- Reviewer guidelines

### Examples & Data

**1. examples/brdf_correction_example.py** (385 lines)
- 4 complete working examples
- Run with: `python examples/brdf_correction_example.py`

**2. data/sample/sample_tropomi_brdf.csv** (31 rows)
- Real TROPOMI observations
- Complete with angle and BRDF data
- Ready for testing

---

## 🚀 Usage Patterns

### Pattern 1: Simple Correction
```python
from src.brdf_correction import brdf_correction

corrected = brdf_correction(
    sun_zenith=df['sza'],
    view_zenith=df['vza'],
    relative_azimuth=df['raa'],
    iso_coefficient=df['iso_r'],
    vol_coefficient=df['vol_r'],
    geo_coefficient=df['geo_r']
)
```

### Pattern 2: Multi-Angle Generation
```python
from src.brdf_correction import apply_multi_angle_correction

df = apply_multi_angle_correction(
    df,
    view_zenith_steps=[0, 20, 40, 60],
    sun_zenith_steps=[30, 45, 60],
    verbose=True
)
```

### Pattern 3: Kernel Inspection
```python
corrected, ross_k, li_k = brdf_correction(
    ...,
    return_kernels=True
)
```

---

## 📊 Module Statistics

| Metric | Value |
|--------|-------|
| **Code Lines** | 918 |
| **Doc Lines** | 1,430 |
| **Total** | 2,348+ |
| **Functions** | 6 main functions |
| **Examples** | 4 complete examples |
| **Files** | 6 code/doc files |
| **Sample Data** | 31 observations |

---

## ✅ Implementation Checklist

- ✅ BRDF kernels (Ross-thick, Li-sparse)
- ✅ Main correction function
- ✅ Multi-angle wrapper
- ✅ Input validation
- ✅ Error handling
- ✅ Type hints on all functions
- ✅ Full docstrings
- ✅ Complete documentation (1430 lines)
- ✅ 4 working examples
- ✅ Sample data provided
- ✅ Backward compatibility
- ✅ Production-ready quality

---

## 🔍 Finding Things

### I want to...

**...understand what BRDF correction does**
→ Read [BRDF_INTEGRATION.md](BRDF_INTEGRATION.md) (10 min)

**...see working code examples**
→ Read [examples/brdf_correction_example.py](examples/brdf_correction_example.py) (10 min)
→ Run `python examples/brdf_correction_example.py` (5 min)

**...understand the theory**
→ Read [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md) - Theory section (20 min)

**...integrate it into my code**
→ Read [BRDF_INTEGRATION.md](BRDF_INTEGRATION.md) - Integration section (15 min)
→ Copy patterns from examples (10 min)

**...understand the implementation details**
→ Read [BRDF_CALL_TRACE.md](BRDF_CALL_TRACE.md) (30 min)
→ Read [src/brdf_correction.py](src/brdf_correction.py) (45 min)

**...test the module**
→ Use [data/sample/sample_tropomi_brdf.csv](data/sample/sample_tropomi_brdf.csv)
→ Run [examples/brdf_correction_example.py](examples/brdf_correction_example.py)

**...fix a problem**
→ Check [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md) - Troubleshooting section

---

## 📚 Reading Order

**For Users (Want to use the code)**:
1. [BRDF_INTEGRATION.md](BRDF_INTEGRATION.md) (5 min)
2. [examples/brdf_correction_example.py](examples/brdf_correction_example.py) (10 min)
3. [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md) - Usage section (15 min)

**For Reviewers (Want to verify correctness)**:
1. [BRDF_DELIVERY_REPORT.md](BRDF_DELIVERY_REPORT.md) (15 min)
2. [src/brdf_correction.py](src/brdf_correction.py) (30 min)
3. [BRDF_CALL_TRACE.md](BRDF_CALL_TRACE.md) (30 min)

**For Developers (Want to extend/modify)**:
1. [BRDF_CALL_TRACE.md](BRDF_CALL_TRACE.md) (30 min)
2. [src/brdf_correction.py](src/brdf_correction.py) (45 min)
3. [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md) - Theory section (25 min)

---

## 🎓 Learning Path

### Beginner (Just starting)
- [ ] Read BRDF_INTEGRATION.md (Overview)
- [ ] Run the examples
- [ ] Test with sample data

### Intermediate (Understanding the code)
- [ ] Read docs/BRDF_GUIDE.md
- [ ] Study examples in detail
- [ ] Try modifying examples

### Advanced (Integration & extension)
- [ ] Read BRDF_CALL_TRACE.md
- [ ] Study src/brdf_correction.py
- [ ] Integrate into your pipeline
- [ ] Extend with custom features

---

## 🤔 FAQ

**Q: Where do I start?**
A: Read [BRDF_INTEGRATION.md](BRDF_INTEGRATION.md) (5 min), then run examples (5 min)

**Q: How do I use this in my code?**
A: See examples in [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md) - Usage section

**Q: What are the input requirements?**
A: See [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md) - Input Data Format section

**Q: How do I fix NaN values?**
A: See [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md) - Troubleshooting section

**Q: How fast is this?**
A: See [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md) - Performance section

**Q: How do I integrate this with my data?**
A: See [BRDF_INTEGRATION.md](BRDF_INTEGRATION.md) - Integration section

---

## 🔗 Cross-References

### Core Module Functions
- `brdf_correction()` → [docs/BRDF_GUIDE.md#basic-usage](docs/BRDF_GUIDE.md)
- `apply_multi_angle_correction()` → [docs/BRDF_GUIDE.md#advanced-usage](docs/BRDF_GUIDE.md)
- `validate_brdf_inputs()` → [BRDF_CALL_TRACE.md#validation](BRDF_CALL_TRACE.md)
- `ross_thick_kernel()` → [docs/BRDF_GUIDE.md#theory](docs/BRDF_GUIDE.md)
- `li_sparse_kernel()` → [docs/BRDF_GUIDE.md#theory](docs/BRDF_GUIDE.md)

### Examples
- Basic correction → [examples/brdf_correction_example.py](examples/brdf_correction_example.py)
- Multi-angle → [examples/brdf_correction_example.py](examples/brdf_correction_example.py)
- Kernel inspection → [examples/brdf_correction_example.py](examples/brdf_correction_example.py)

### Documentation
- Theory → [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md)
- Usage → [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md)
- Integration → [BRDF_INTEGRATION.md](BRDF_INTEGRATION.md)
- Architecture → [BRDF_CALL_TRACE.md](BRDF_CALL_TRACE.md)

---

## 📞 Support

**For questions about**:
- **Theory**: See [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md) - Theory section
- **Usage**: See [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md) - Usage Guide section
- **Errors**: See [docs/BRDF_GUIDE.md](docs/BRDF_GUIDE.md) - Troubleshooting section
- **Integration**: See [BRDF_INTEGRATION.md](BRDF_INTEGRATION.md)
- **Implementation**: See [BRDF_CALL_TRACE.md](BRDF_CALL_TRACE.md)

---

**Status**: ✅ Complete and ready for use
**Version**: 1.0
**Last Updated**: February 10, 2026
