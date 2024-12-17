#include "Common/Utils.h"

namespace KernelCodeGen {


namespace tools {
mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype) {
  if(dtype == "float32") return builder.getF32Type();
  if(dtype == "float64") return builder.getF64Type();
  if(dtype == "float16") return builder.getF16Type();
  if(dtype == "int64") return builder.getIntegerType(64);
  if(dtype == "int32") return builder.getIntegerType(32);
  if(dtype == "int16") return builder.getIntegerType(16);
  if(dtype == "index") return builder.getIndexType();
  if(dtype == "bool") return builder.getIntegerType(1);
  assert(false && "getDType:: Unsupported Type!");
  return nullptr;
}

std::string getenv(const char *name) {
    const char *cstr = std::getenv(name);
    if (!cstr)
        return "";
    std::string result(cstr);
    return result;
}


std::string KcgDtypeToStr(KcgDtype type){
  switch (type){
    case KcgDtype::float8   : return "";break;
    case KcgDtype::float16  : return "float16";break;
    case KcgDtype::float32  : return "float32";break;
    case KcgDtype::float64  : return "float64";break;
    case KcgDtype::float128 : return "";break;
    case KcgDtype::int8     : return "";break;
    case KcgDtype::int16    : return "int16";break;
    case KcgDtype::int32    : return "int32";break;
    case KcgDtype::int64    : return "int64";break;
  default:
    assert(false && "KcgDtypeToStr::invalid type!");
    break;
  }
  return "";
}

std::string typeToStr(mlir::Type type) {
  if(type.isa<mlir::Float16Type>()) return {"float16"};
  if(type.isa<mlir::Float32Type>()) return {"float32"};
  if(type.isa<mlir::Float64Type>()) return {"float64"};
  if(auto int_type = type.dyn_cast<mlir::IntegerType>()) {
    if (int_type.getWidth() == 1) return {"bool"};
    else if (int_type.getWidth() == 16) return {"int16"};
    else if (int_type.getWidth() == 32) return {"int32"};
    else if (int_type.getWidth() == 64) return {"int64"};
  }
  if(type.isa<mlir::IndexType>()) return {"index"};
  assert(false && "not supported type!");
  return "";
}

void opSetAttr(mlir::Operation* op, const std::string& name, const std::string& val){
  assert(op != nullptr && "opSetAttr::nullptr error!");
  mlir::OpBuilder b(op->getContext());
  op->setAttr(name,b.getStringAttr(val)); 
}

bool isOpAttrEqualToString(mlir::Operation* op, const std::string& name, const std::string& expectedvalue){
  assert(op != nullptr && "isOpAttrEqualToString::nullptr error!");
  if(op->hasAttr(name)){
    auto attr = mlir::dyn_cast<mlir::StringAttr>(op->getAttr(name));
    if(attr.getValue() == expectedvalue){
      return true;
    }
  }
  return false;
}

std::string KcgKernelTypeToString(KcgKernelType type){
  switch (type)
  {
  case KcgKernelType::matmul:
    return "matmul"; break;
  case KcgKernelType::conv2d:
    return "conv2d"; break;
  case KcgKernelType::poolmax:
    return "poolmax"; break;
  default:
    assert(false && "KcgKernelTypeToString::invalid type!");break;
  }
  return "Unknown";
}

void _opSetDescription(mlir::Operation* op, const std::string& attrValue){
  mlir::OpBuilder b(op->getContext());
  op->setAttr(AttrDescription, b.getStringAttr(attrValue));
}


}  // namespace tools
}  // namespace KernelCodeGen