//  === after secondLowering =====


// llvm.func @__ockl_printf_append_string_n(i64, !llvm.ptr, i64, i32) -> i64
// llvm.func @__ockl_printf_append_args(i64, i32, i64, i64, i64, i64, i64, i64, i64, i32) -> i64
// llvm.func @__ockl_printf_begin(i64) -> i64
// llvm.mlir.global internal constant @printfFormat_0("pid (%u, %u, %u) =======device=======\0A\00") {addr_space = 0 : i32}
// llvm.mlir.global internal constant @printfPrefix_0(" =======device=======") {addr_space = 0 : i32}

// %cc39 = llvm.mlir.constant(39 : i64) : i64
// %cc20 = llvm.mlir.constant(0 : i32) : i32
// %cc0 = llvm.mlir.constant(0 : i64) : i64
// %cc1 = llvm.mlir.constant(1 : i32) : i32
// %cc3 = llvm.mlir.constant(3 : i32) : i32
// %r0 = llvm.mlir.addressof @printfFormat_0 : !llvm.ptr<array<39 x i8>>
// %r1 = llvm.getelementptr %r0[0, 0] : (!llvm.ptr<array<39 x i8>>) -> !llvm.ptr<i8>
// %r2 = llvm.call @__ockl_printf_begin(%cc0) : (i64) -> i64
// %r3 = llvm.bitcast %r1 : !llvm.ptr<i8> to !llvm.ptr
// %r4 = llvm.call @__ockl_printf_append_string_n(%r2, %r3, %cc39, %cc20) : (i64, !llvm.ptr, i64, i32) -> i64
// %cll = llvm.call @__ockl_printf_append_args(%r4, %cc3, %cc0, %cc0, %cc0, %cc0, %cc0, %cc0, %cc0, %cc1) : (i64, i32, i64, i64, i64, i64, i64, i64, i64, i32) -> i64




module attributes {kcg.externLibs = {library_0 = "/home/xushilong/triton_rocm/triton/python/triton/third_party/hip/lib/bitcode/opencl.bc", library_1 = "/home/xushilong/triton_rocm/triton/python/triton/third_party/hip/lib/bitcode/ocml.bc", library_2 = "/home/xushilong/triton_rocm/triton/python/triton/third_party/hip/lib/bitcode/ockl.bc", library_3 = "/home/xushilong/triton_rocm/triton/python/triton/third_party/hip/lib/bitcode/oclc_finite_only_off.bc", library_4 = "/home/xushilong/triton_rocm/triton/python/triton/third_party/hip/lib/bitcode/oclc_daz_opt_on.bc", library_5 = "/home/xushilong/triton_rocm/triton/python/triton/third_party/hip/lib/bitcode/oclc_correctly_rounded_sqrt_on.bc", library_6 = "/home/xushilong/triton_rocm/triton/python/triton/third_party/hip/lib/bitcode/oclc_unsafe_math_off.bc", library_7 = "/home/xushilong/triton_rocm/triton/python/triton/third_party/hip/lib/bitcode/oclc_wavefrontsize64_on.bc", library_8 = "/home/xushilong/triton_rocm/triton/python/triton/third_party/hip/lib/bitcode/oclc_abi_version_400.bc", library_9 = "/home/xushilong/triton_rocm/triton/python/triton/third_party/hip/lib/bitcode/oclc_isa_version_906.bc"}} {
  llvm.func @__ockl_printf_append_string_n(i64, !llvm.ptr, i64, i32) -> i64
  llvm.func @__ockl_printf_append_args(i64, i32, i64, i64, i64, i64, i64, i64, i64, i32) -> i64
  llvm.func @__ockl_printf_begin(i64) -> i64
  llvm.mlir.global internal constant @printfFormat_0("pid (%u, %u, %u) =======device=======\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @printfPrefix_0(" =======device=======") {addr_space = 0 : i32}
  llvm.mlir.global external @kcg_shm0() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x f32>
  llvm.func @GEMM_mnk1024x1024x1024_f32f32f32_TTmn4x4_BTmnk64x64x16_BLmn4x1_WLmn4x16(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {func.block.dim = array<i32: 16, 16>, func.grid.dim = array<i32: 16, 16>, func.op.name = "Matmul", func.state = "gpu", nvvm.kernel = 1 : i32 } 
  {
    %0 = llvm.mlir.constant(3 : index) : i32
    %1 = llvm.mlir.constant(8 : index) : i32
    %2 = llvm.mlir.constant(2 : index) : i32
    %3 = llvm.mlir.constant(4 : index) : i32
    %4 = llvm.mlir.constant(960 : index) : i32
    %5 = llvm.mlir.constant(11 : index) : i32
    %6 = llvm.mlir.constant(10 : index) : i32
    %7 = llvm.mlir.constant(9 : index) : i32
    %8 = llvm.mlir.constant(19 : index) : i32
    %9 = llvm.mlir.constant(18 : index) : i32
    %10 = llvm.mlir.constant(17 : index) : i32
    %11 = llvm.mlir.constant(16 : index) : i32
    %12 = llvm.mlir.constant(256 : index) : i32
    %13 = llvm.mlir.constant(14 : index) : i32
    %14 = llvm.mlir.constant(992 : index) : i32
    %15 = llvm.mlir.constant(2048 : index) : i32
    %16 = llvm.mlir.constant(64 : index) : i32
    %17 = llvm.mlir.constant(-1 : index) : i32
    %18 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %19 = llvm.mlir.constant(40 : index) : i32
    %20 = llvm.mlir.constant(1 : index) : i32
    %21 = llvm.mlir.constant(1024 : index) : i32
    %22 = llvm.mlir.constant(0 : index) : i32
    %23 = rocdl.workgroup.id.y {range = array<i32: 0, 16>} : i32
    %24 = rocdl.workgroup.id.x {range = array<i32: 0, 16>} : i32
    %25 = rocdl.workitem.id.y {range = array<i32: 0, 16>} : i32
    %26 = rocdl.workitem.id.x {range = array<i32: 0, 16>} : i32

    %27 = llvm.mlir.addressof @kcg_shm0 : !llvm.ptr<3>
    %28 = llvm.alloca %19 x f32 {alignment = 16 : i64} : (i32) -> !llvm.ptr<5>
    %29 = llvm.getelementptr %28[24] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %29 : f32, !llvm.ptr<5>
    %30 = llvm.getelementptr %28[25] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %30 : f32, !llvm.ptr<5>
    %31 = llvm.getelementptr %28[26] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %31 : f32, !llvm.ptr<5>
    %32 = llvm.getelementptr %28[27] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %32 : f32, !llvm.ptr<5>
    %33 = llvm.getelementptr %28[28] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %33 : f32, !llvm.ptr<5>
    %34 = llvm.getelementptr %28[29] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %34 : f32, !llvm.ptr<5>
    %35 = llvm.getelementptr %28[30] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %35 : f32, !llvm.ptr<5>
    %36 = llvm.getelementptr %28[31] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %36 : f32, !llvm.ptr<5>
    %37 = llvm.getelementptr %28[32] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %37 : f32, !llvm.ptr<5>
    %38 = llvm.getelementptr %28[33] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %38 : f32, !llvm.ptr<5>
    %39 = llvm.getelementptr %28[34] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %39 : f32, !llvm.ptr<5>
    %40 = llvm.getelementptr %28[35] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %40 : f32, !llvm.ptr<5>
    %41 = llvm.getelementptr %28[36] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %41 : f32, !llvm.ptr<5>
    %42 = llvm.getelementptr %28[37] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %42 : f32, !llvm.ptr<5>
    %43 = llvm.getelementptr %28[38] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %43 : f32, !llvm.ptr<5>
    %44 = llvm.getelementptr %28[39] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %18, %44 : f32, !llvm.ptr<5>
    %45 = llvm.mul %25, %3  : i32
    %46 = llvm.icmp "slt" %26, %22 : i32
    %47 = llvm.sub %17, %26  : i32
    %48 = llvm.select %46, %47, %26 : i1, i32
    %49 = llvm.sdiv %48, %3  : i32
    %50 = llvm.sub %17, %49  : i32
    %51 = llvm.select %46, %50, %49 : i1, i32
    %52 = llvm.add %45, %51  : i32
    %53 = llvm.mul %23, %16  : i32
    %54 = llvm.add %52, %53  : i32
    %55 = llvm.srem %26, %3  : i32
    %56 = llvm.icmp "slt" %55, %22 : i32
    %57 = llvm.add %55, %3  : i32
    %58 = llvm.select %56, %57, %55 : i1, i32
    %59 = llvm.mul %58, %3  : i32
    %60 = llvm.mul %54, %21  : i32
    %61 = llvm.add %60, %59  : i32
    %62 = llvm.getelementptr %arg0[%61] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %63 = llvm.load %62 {alignment = 4 : i64} : !llvm.ptr<1> -> vector<4xf32>
    llvm.store %63, %28 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    %64 = llvm.sdiv %48, %11  : i32
    %65 = llvm.sub %17, %64  : i32
    %66 = llvm.select %46, %65, %64 : i1, i32
    %67 = llvm.add %25, %66  : i32
    %68 = llvm.srem %26, %11  : i32
    %69 = llvm.icmp "slt" %68, %22 : i32
    %70 = llvm.add %68, %11  : i32
    %71 = llvm.select %69, %70, %68 : i1, i32
    %72 = llvm.mul %71, %3  : i32
    %73 = llvm.mul %24, %16  : i32
    %74 = llvm.add %72, %73  : i32
    %75 = llvm.mul %67, %21  : i32
    %76 = llvm.add %75, %74  : i32
    %77 = llvm.getelementptr %arg1[%76] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %78 = llvm.load %77 {alignment = 4 : i64} : !llvm.ptr<1> -> vector<4xf32>
    %79 = llvm.getelementptr %28[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %78, %79 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    %80 = llvm.load %28 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<1xf32>
    %81 = llvm.mul %58, %12  : i32
    %82 = llvm.add %81, %52  : i32
    %83 = llvm.getelementptr %27[%82] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.store %80, %83 {alignment = 4 : i64} : vector<1xf32>, !llvm.ptr<3>
    %84 = llvm.getelementptr %28[1] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %85 = llvm.load %84 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<1xf32>
    %86 = llvm.add %59, %20  : i32
    %87 = llvm.mul %86, %16  : i32
    %88 = llvm.add %87, %52  : i32
    %89 = llvm.getelementptr %27[%88] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.store %85, %89 {alignment = 4 : i64} : vector<1xf32>, !llvm.ptr<3>
    %90 = llvm.getelementptr %28[2] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %91 = llvm.load %90 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<1xf32>
    %92 = llvm.add %59, %2  : i32
    %93 = llvm.mul %92, %16  : i32
    %94 = llvm.add %93, %52  : i32
    %95 = llvm.getelementptr %27[%94] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.store %91, %95 {alignment = 4 : i64} : vector<1xf32>, !llvm.ptr<3>
    %96 = llvm.getelementptr %28[3] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %97 = llvm.load %96 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<1xf32>
    %98 = llvm.add %59, %0  : i32
    %99 = llvm.mul %98, %16  : i32
    %100 = llvm.add %99, %52  : i32
    %101 = llvm.getelementptr %27[%100] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.store %97, %101 {alignment = 4 : i64} : vector<1xf32>, !llvm.ptr<3>
    %102 = llvm.load %79 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<4xf32>
    %103 = llvm.mul %67, %16  : i32
    %104 = llvm.add %103, %72  : i32
    %105 = llvm.add %104, %15  : i32
    %106 = llvm.getelementptr %27[%105] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.store %102, %106 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<3>
    
%cc39 = llvm.mlir.constant(39 : i64) : i64
%cc20 = llvm.mlir.constant(0 : i32) : i32
%cc0 = llvm.mlir.constant(0 : i64) : i64
%cc1 = llvm.mlir.constant(1 : i32) : i32
%cc3 = llvm.mlir.constant(3 : i32) : i32
%r0 = llvm.mlir.addressof @printfFormat_0 : !llvm.ptr<array<39 x i8>>
%r1 = llvm.getelementptr %r0[0, 0] : (!llvm.ptr<array<39 x i8>>) -> !llvm.ptr<i8>
%r2 = llvm.call @__ockl_printf_begin(%cc0) : (i64) -> i64
%r3 = llvm.bitcast %r1 : !llvm.ptr<i8> to !llvm.ptr
%r4 = llvm.call @__ockl_printf_append_string_n(%r2, %r3, %cc39, %cc20) : (i64, !llvm.ptr, i64, i32) -> i64
%cll = llvm.call @__ockl_printf_append_args(%r4, %cc3, %cc0, %cc0, %cc0, %cc0, %cc0, %cc0, %cc0, %cc1) : (i64, i32, i64, i64, i64, i64, i64, i64, i64, i32) -> i64

    rocdl.barrier
    %107 = llvm.mul %25, %11  : i32
    %108 = llvm.add %107, %26  : i32
    %109 = llvm.srem %108, %16  : i32
    %110 = llvm.icmp "slt" %109, %22 : i32
    %111 = llvm.add %109, %16  : i32
    %112 = llvm.select %110, %111, %109 : i1, i32
    %113 = llvm.icmp "slt" %112, %22 : i32
    %114 = llvm.sub %17, %112  : i32
    %115 = llvm.select %113, %114, %112 : i1, i32
    %116 = llvm.sdiv %115, %11  : i32
    %117 = llvm.sub %17, %116  : i32
    %118 = llvm.select %113, %117, %116 : i1, i32
    %119 = llvm.icmp "slt" %108, %22 : i32
    %120 = llvm.sub %17, %108  : i32
    %121 = llvm.select %119, %120, %108 : i1, i32
    %122 = llvm.sdiv %121, %16  : i32
    %123 = llvm.sub %17, %122  : i32
    %124 = llvm.select %119, %123, %122 : i1, i32
    %125 = llvm.mul %124, %3  : i32
    %126 = llvm.add %118, %125  : i32
    %127 = llvm.mul %126, %3  : i32
    %128 = llvm.getelementptr %27[%127] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %129 = llvm.load %128 {alignment = 4 : i64} : !llvm.ptr<3> -> vector<4xf32>
    %130 = llvm.getelementptr %28[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %129, %130 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    %131 = llvm.add %72, %15  : i32
    %132 = llvm.getelementptr %27[%131] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %133 = llvm.load %132 {alignment = 4 : i64} : !llvm.ptr<3> -> vector<4xf32>
    %134 = llvm.getelementptr %28[16] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %133, %134 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    llvm.br ^bb1(%22 : i32)
  ^bb1(%135: i32):  // 2 preds: ^bb0, ^bb9
    %136 = llvm.icmp "slt" %135, %21 : i32
    llvm.cond_br %136, ^bb2, ^bb10
  ^bb2:  // pred: ^bb1
    %137 = llvm.sub %14, %135  : i32
    %138 = llvm.icmp "sge" %137, %22 : i32
    llvm.cond_br %138, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %139 = llvm.add %135, %11  : i32
    %140 = llvm.add %59, %139  : i32
    %141 = llvm.add %60, %140  : i32
    %142 = llvm.getelementptr %arg0[%141] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %143 = llvm.load %142 {alignment = 4 : i64} : !llvm.ptr<1> -> vector<4xf32>
    llvm.store %143, %28 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    %144 = llvm.add %67, %139  : i32
    %145 = llvm.mul %144, %21  : i32
    %146 = llvm.add %145, %74  : i32
    %147 = llvm.getelementptr %arg1[%146] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %148 = llvm.load %147 {alignment = 4 : i64} : !llvm.ptr<1> -> vector<4xf32>
    llvm.store %148, %79 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %149 = llvm.icmp "slt" %135, %22 : i32
    %150 = llvm.sub %17, %135  : i32
    %151 = llvm.select %149, %150, %135 : i1, i32
    %152 = llvm.sdiv %151, %11  : i32
    %153 = llvm.sub %17, %152  : i32
    %154 = llvm.select %149, %153, %152 : i1, i32
    %155 = llvm.srem %154, %2  : i32
    %156 = llvm.icmp "slt" %155, %22 : i32
    %157 = llvm.add %155, %2  : i32
    %158 = llvm.select %156, %157, %155 : i1, i32
    %159 = llvm.mul %158, %21  : i32
    llvm.br ^bb5(%22 : i32)
  ^bb5(%160: i32):  // 2 preds: ^bb4, ^bb6
    %161 = llvm.icmp "slt" %160, %13 : i32
    llvm.cond_br %161, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %162 = llvm.add %160, %20  : i32
    %163 = llvm.mul %162, %16  : i32
    %164 = llvm.add %159, %163  : i32
    %165 = llvm.add %164, %127  : i32
    %166 = llvm.getelementptr %27[%165] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %167 = llvm.load %166 {alignment = 4 : i64} : !llvm.ptr<3> -> vector<4xf32>
    %168 = llvm.srem %162, %2  : i32
    %169 = llvm.icmp "slt" %168, %22 : i32
    %170 = llvm.add %168, %2  : i32
    %171 = llvm.select %169, %170, %168 : i1, i32
    %172 = llvm.mul %171, %3  : i32
    %173 = llvm.add %172, %1  : i32
    %174 = llvm.getelementptr %28[%173] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    llvm.store %167, %174 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    %175 = llvm.add %164, %72  : i32
    %176 = llvm.add %175, %15  : i32
    %177 = llvm.getelementptr %27[%176] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %178 = llvm.load %177 {alignment = 4 : i64} : !llvm.ptr<3> -> vector<4xf32>
    %179 = llvm.add %172, %11  : i32
    %180 = llvm.getelementptr %28[%179] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    llvm.store %178, %180 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    %181 = llvm.load %29 : !llvm.ptr<5> -> f32
    %182 = llvm.srem %160, %2  : i32
    %183 = llvm.icmp "slt" %182, %22 : i32
    %184 = llvm.add %182, %2  : i32
    %185 = llvm.select %183, %184, %182 : i1, i32
    %186 = llvm.mul %185, %3  : i32
    %187 = llvm.add %186, %1  : i32
    %188 = llvm.getelementptr %28[%187] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %189 = llvm.load %188 : !llvm.ptr<5> -> f32
    %190 = llvm.add %186, %11  : i32
    %191 = llvm.getelementptr %28[%190] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %192 = llvm.load %191 : !llvm.ptr<5> -> f32
    %193 = llvm.fmul %189, %192  : f32
    %194 = llvm.fadd %193, %181  : f32
    llvm.store %194, %29 : f32, !llvm.ptr<5>
    %195 = llvm.load %30 : !llvm.ptr<5> -> f32
    %196 = llvm.load %188 : !llvm.ptr<5> -> f32
    %197 = llvm.add %186, %10  : i32
    %198 = llvm.getelementptr %28[%197] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %199 = llvm.load %198 : !llvm.ptr<5> -> f32
    %200 = llvm.fmul %196, %199  : f32
    %201 = llvm.fadd %200, %195  : f32
    llvm.store %201, %30 : f32, !llvm.ptr<5>
    %202 = llvm.load %31 : !llvm.ptr<5> -> f32
    %203 = llvm.load %188 : !llvm.ptr<5> -> f32
    %204 = llvm.add %186, %9  : i32
    %205 = llvm.getelementptr %28[%204] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %206 = llvm.load %205 : !llvm.ptr<5> -> f32
    %207 = llvm.fmul %203, %206  : f32
    %208 = llvm.fadd %207, %202  : f32
    llvm.store %208, %31 : f32, !llvm.ptr<5>
    %209 = llvm.load %32 : !llvm.ptr<5> -> f32
    %210 = llvm.load %188 : !llvm.ptr<5> -> f32
    %211 = llvm.add %186, %8  : i32
    %212 = llvm.getelementptr %28[%211] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %213 = llvm.load %212 : !llvm.ptr<5> -> f32
    %214 = llvm.fmul %210, %213  : f32
    %215 = llvm.fadd %214, %209  : f32
    llvm.store %215, %32 : f32, !llvm.ptr<5>
    %216 = llvm.load %33 : !llvm.ptr<5> -> f32
    %217 = llvm.add %186, %7  : i32
    %218 = llvm.getelementptr %28[%217] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %219 = llvm.load %218 : !llvm.ptr<5> -> f32
    %220 = llvm.load %191 : !llvm.ptr<5> -> f32
    %221 = llvm.fmul %219, %220  : f32
    %222 = llvm.fadd %221, %216  : f32
    llvm.store %222, %33 : f32, !llvm.ptr<5>
    %223 = llvm.load %34 : !llvm.ptr<5> -> f32
    %224 = llvm.load %218 : !llvm.ptr<5> -> f32
    %225 = llvm.load %198 : !llvm.ptr<5> -> f32
    %226 = llvm.fmul %224, %225  : f32
    %227 = llvm.fadd %226, %223  : f32
    llvm.store %227, %34 : f32, !llvm.ptr<5>
    %228 = llvm.load %35 : !llvm.ptr<5> -> f32
    %229 = llvm.load %218 : !llvm.ptr<5> -> f32
    %230 = llvm.load %205 : !llvm.ptr<5> -> f32
    %231 = llvm.fmul %229, %230  : f32
    %232 = llvm.fadd %231, %228  : f32
    llvm.store %232, %35 : f32, !llvm.ptr<5>
    %233 = llvm.load %36 : !llvm.ptr<5> -> f32
    %234 = llvm.load %218 : !llvm.ptr<5> -> f32
    %235 = llvm.load %212 : !llvm.ptr<5> -> f32
    %236 = llvm.fmul %234, %235  : f32
    %237 = llvm.fadd %236, %233  : f32
    llvm.store %237, %36 : f32, !llvm.ptr<5>
    %238 = llvm.load %37 : !llvm.ptr<5> -> f32
    %239 = llvm.add %186, %6  : i32
    %240 = llvm.getelementptr %28[%239] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %241 = llvm.load %240 : !llvm.ptr<5> -> f32
    %242 = llvm.load %191 : !llvm.ptr<5> -> f32
    %243 = llvm.fmul %241, %242  : f32
    %244 = llvm.fadd %243, %238  : f32
    llvm.store %244, %37 : f32, !llvm.ptr<5>
    %245 = llvm.load %38 : !llvm.ptr<5> -> f32
    %246 = llvm.load %240 : !llvm.ptr<5> -> f32
    %247 = llvm.load %198 : !llvm.ptr<5> -> f32
    %248 = llvm.fmul %246, %247  : f32
    %249 = llvm.fadd %248, %245  : f32
    llvm.store %249, %38 : f32, !llvm.ptr<5>
    %250 = llvm.load %39 : !llvm.ptr<5> -> f32
    %251 = llvm.load %240 : !llvm.ptr<5> -> f32
    %252 = llvm.load %205 : !llvm.ptr<5> -> f32
    %253 = llvm.fmul %251, %252  : f32
    %254 = llvm.fadd %253, %250  : f32
    llvm.store %254, %39 : f32, !llvm.ptr<5>
    %255 = llvm.load %40 : !llvm.ptr<5> -> f32
    %256 = llvm.load %240 : !llvm.ptr<5> -> f32
    %257 = llvm.load %212 : !llvm.ptr<5> -> f32
    %258 = llvm.fmul %256, %257  : f32
    %259 = llvm.fadd %258, %255  : f32
    llvm.store %259, %40 : f32, !llvm.ptr<5>
    %260 = llvm.load %41 : !llvm.ptr<5> -> f32
    %261 = llvm.add %186, %5  : i32
    %262 = llvm.getelementptr %28[%261] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %263 = llvm.load %262 : !llvm.ptr<5> -> f32
    %264 = llvm.load %191 : !llvm.ptr<5> -> f32
    %265 = llvm.fmul %263, %264  : f32
    %266 = llvm.fadd %265, %260  : f32
    llvm.store %266, %41 : f32, !llvm.ptr<5>
    %267 = llvm.load %42 : !llvm.ptr<5> -> f32
    %268 = llvm.load %262 : !llvm.ptr<5> -> f32
    %269 = llvm.load %198 : !llvm.ptr<5> -> f32
    %270 = llvm.fmul %268, %269  : f32
    %271 = llvm.fadd %270, %267  : f32
    llvm.store %271, %42 : f32, !llvm.ptr<5>
    %272 = llvm.load %43 : !llvm.ptr<5> -> f32
    %273 = llvm.load %262 : !llvm.ptr<5> -> f32
    %274 = llvm.load %205 : !llvm.ptr<5> -> f32
    %275 = llvm.fmul %273, %274  : f32
    %276 = llvm.fadd %275, %272  : f32
    llvm.store %276, %43 : f32, !llvm.ptr<5>
    %277 = llvm.load %44 : !llvm.ptr<5> -> f32
    %278 = llvm.load %262 : !llvm.ptr<5> -> f32
    %279 = llvm.load %212 : !llvm.ptr<5> -> f32
    %280 = llvm.fmul %278, %279  : f32
    %281 = llvm.fadd %280, %277  : f32
    llvm.store %281, %44 : f32, !llvm.ptr<5>
    %282 = llvm.add %160, %2  : i32
    %283 = llvm.mul %282, %16  : i32
    %284 = llvm.add %159, %283  : i32
    %285 = llvm.add %284, %127  : i32
    %286 = llvm.getelementptr %27[%285] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %287 = llvm.load %286 {alignment = 4 : i64} : !llvm.ptr<3> -> vector<4xf32>
    %288 = llvm.srem %282, %2  : i32
    %289 = llvm.icmp "slt" %288, %22 : i32
    %290 = llvm.add %288, %2  : i32
    %291 = llvm.select %289, %290, %288 : i1, i32
    %292 = llvm.mul %291, %3  : i32
    %293 = llvm.add %292, %1  : i32
    %294 = llvm.getelementptr %28[%293] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    llvm.store %287, %294 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    %295 = llvm.add %284, %72  : i32
    %296 = llvm.add %295, %15  : i32
    %297 = llvm.getelementptr %27[%296] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %298 = llvm.load %297 {alignment = 4 : i64} : !llvm.ptr<3> -> vector<4xf32>
    %299 = llvm.add %292, %11  : i32
    %300 = llvm.getelementptr %28[%299] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    llvm.store %298, %300 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    %301 = llvm.load %29 : !llvm.ptr<5> -> f32
    %302 = llvm.load %174 : !llvm.ptr<5> -> f32
    %303 = llvm.load %180 : !llvm.ptr<5> -> f32
    %304 = llvm.fmul %302, %303  : f32
    %305 = llvm.fadd %304, %301  : f32
    llvm.store %305, %29 : f32, !llvm.ptr<5>
    %306 = llvm.load %30 : !llvm.ptr<5> -> f32
    %307 = llvm.load %174 : !llvm.ptr<5> -> f32
    %308 = llvm.add %172, %10  : i32
    %309 = llvm.getelementptr %28[%308] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %310 = llvm.load %309 : !llvm.ptr<5> -> f32
    %311 = llvm.fmul %307, %310  : f32
    %312 = llvm.fadd %311, %306  : f32
    llvm.store %312, %30 : f32, !llvm.ptr<5>
    %313 = llvm.load %31 : !llvm.ptr<5> -> f32
    %314 = llvm.load %174 : !llvm.ptr<5> -> f32
    %315 = llvm.add %172, %9  : i32
    %316 = llvm.getelementptr %28[%315] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %317 = llvm.load %316 : !llvm.ptr<5> -> f32
    %318 = llvm.fmul %314, %317  : f32
    %319 = llvm.fadd %318, %313  : f32
    llvm.store %319, %31 : f32, !llvm.ptr<5>
    %320 = llvm.load %32 : !llvm.ptr<5> -> f32
    %321 = llvm.load %174 : !llvm.ptr<5> -> f32
    %322 = llvm.add %172, %8  : i32
    %323 = llvm.getelementptr %28[%322] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %324 = llvm.load %323 : !llvm.ptr<5> -> f32
    %325 = llvm.fmul %321, %324  : f32
    %326 = llvm.fadd %325, %320  : f32
    llvm.store %326, %32 : f32, !llvm.ptr<5>
    %327 = llvm.load %33 : !llvm.ptr<5> -> f32
    %328 = llvm.add %172, %7  : i32
    %329 = llvm.getelementptr %28[%328] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %330 = llvm.load %329 : !llvm.ptr<5> -> f32
    %331 = llvm.load %180 : !llvm.ptr<5> -> f32
    %332 = llvm.fmul %330, %331  : f32
    %333 = llvm.fadd %332, %327  : f32
    llvm.store %333, %33 : f32, !llvm.ptr<5>
    %334 = llvm.load %34 : !llvm.ptr<5> -> f32
    %335 = llvm.load %329 : !llvm.ptr<5> -> f32
    %336 = llvm.load %309 : !llvm.ptr<5> -> f32
    %337 = llvm.fmul %335, %336  : f32
    %338 = llvm.fadd %337, %334  : f32
    llvm.store %338, %34 : f32, !llvm.ptr<5>
    %339 = llvm.load %35 : !llvm.ptr<5> -> f32
    %340 = llvm.load %329 : !llvm.ptr<5> -> f32
    %341 = llvm.load %316 : !llvm.ptr<5> -> f32
    %342 = llvm.fmul %340, %341  : f32
    %343 = llvm.fadd %342, %339  : f32
    llvm.store %343, %35 : f32, !llvm.ptr<5>
    %344 = llvm.load %36 : !llvm.ptr<5> -> f32
    %345 = llvm.load %329 : !llvm.ptr<5> -> f32
    %346 = llvm.load %323 : !llvm.ptr<5> -> f32
    %347 = llvm.fmul %345, %346  : f32
    %348 = llvm.fadd %347, %344  : f32
    llvm.store %348, %36 : f32, !llvm.ptr<5>
    %349 = llvm.load %37 : !llvm.ptr<5> -> f32
    %350 = llvm.add %172, %6  : i32
    %351 = llvm.getelementptr %28[%350] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %352 = llvm.load %351 : !llvm.ptr<5> -> f32
    %353 = llvm.load %180 : !llvm.ptr<5> -> f32
    %354 = llvm.fmul %352, %353  : f32
    %355 = llvm.fadd %354, %349  : f32
    llvm.store %355, %37 : f32, !llvm.ptr<5>
    %356 = llvm.load %38 : !llvm.ptr<5> -> f32
    %357 = llvm.load %351 : !llvm.ptr<5> -> f32
    %358 = llvm.load %309 : !llvm.ptr<5> -> f32
    %359 = llvm.fmul %357, %358  : f32
    %360 = llvm.fadd %359, %356  : f32
    llvm.store %360, %38 : f32, !llvm.ptr<5>
    %361 = llvm.load %39 : !llvm.ptr<5> -> f32
    %362 = llvm.load %351 : !llvm.ptr<5> -> f32
    %363 = llvm.load %316 : !llvm.ptr<5> -> f32
    %364 = llvm.fmul %362, %363  : f32
    %365 = llvm.fadd %364, %361  : f32
    llvm.store %365, %39 : f32, !llvm.ptr<5>
    %366 = llvm.load %40 : !llvm.ptr<5> -> f32
    %367 = llvm.load %351 : !llvm.ptr<5> -> f32
    %368 = llvm.load %323 : !llvm.ptr<5> -> f32
    %369 = llvm.fmul %367, %368  : f32
    %370 = llvm.fadd %369, %366  : f32
    llvm.store %370, %40 : f32, !llvm.ptr<5>
    %371 = llvm.load %41 : !llvm.ptr<5> -> f32
    %372 = llvm.add %172, %5  : i32
    %373 = llvm.getelementptr %28[%372] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
    %374 = llvm.load %373 : !llvm.ptr<5> -> f32
    %375 = llvm.load %180 : !llvm.ptr<5> -> f32
    %376 = llvm.fmul %374, %375  : f32
    %377 = llvm.fadd %376, %371  : f32
    llvm.store %377, %41 : f32, !llvm.ptr<5>
    %378 = llvm.load %42 : !llvm.ptr<5> -> f32
    %379 = llvm.load %373 : !llvm.ptr<5> -> f32
    %380 = llvm.load %309 : !llvm.ptr<5> -> f32
    %381 = llvm.fmul %379, %380  : f32
    %382 = llvm.fadd %381, %378  : f32
    llvm.store %382, %42 : f32, !llvm.ptr<5>
    %383 = llvm.load %43 : !llvm.ptr<5> -> f32
    %384 = llvm.load %373 : !llvm.ptr<5> -> f32
    %385 = llvm.load %316 : !llvm.ptr<5> -> f32
    %386 = llvm.fmul %384, %385  : f32
    %387 = llvm.fadd %386, %383  : f32
    llvm.store %387, %43 : f32, !llvm.ptr<5>
    %388 = llvm.load %44 : !llvm.ptr<5> -> f32
    %389 = llvm.load %373 : !llvm.ptr<5> -> f32
    %390 = llvm.load %323 : !llvm.ptr<5> -> f32
    %391 = llvm.fmul %389, %390  : f32
    %392 = llvm.fadd %391, %388  : f32
    llvm.store %392, %44 : f32, !llvm.ptr<5>
    llvm.br ^bb5(%282 : i32)
  ^bb7:  // pred: ^bb5
    %393 = llvm.add %159, %4  : i32
    %394 = llvm.add %393, %127  : i32
    %395 = llvm.getelementptr %27[%394] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %396 = llvm.load %395 {alignment = 4 : i64} : !llvm.ptr<3> -> vector<4xf32>
    %397 = llvm.getelementptr %28[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %396, %397 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    %398 = llvm.add %393, %72  : i32
    %399 = llvm.add %398, %15  : i32
    %400 = llvm.getelementptr %27[%399] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %401 = llvm.load %400 {alignment = 4 : i64} : !llvm.ptr<3> -> vector<4xf32>
    %402 = llvm.getelementptr %28[20] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    llvm.store %401, %402 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    %403 = llvm.load %29 : !llvm.ptr<5> -> f32
    %404 = llvm.load %130 : !llvm.ptr<5> -> f32
    %405 = llvm.load %134 : !llvm.ptr<5> -> f32
    %406 = llvm.fmul %404, %405  : f32
    %407 = llvm.fadd %406, %403  : f32
    llvm.store %407, %29 : f32, !llvm.ptr<5>
    %408 = llvm.load %30 : !llvm.ptr<5> -> f32
    %409 = llvm.load %130 : !llvm.ptr<5> -> f32
    %410 = llvm.getelementptr %28[17] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %411 = llvm.load %410 : !llvm.ptr<5> -> f32
    %412 = llvm.fmul %409, %411  : f32
    %413 = llvm.fadd %412, %408  : f32
    llvm.store %413, %30 : f32, !llvm.ptr<5>
    %414 = llvm.load %31 : !llvm.ptr<5> -> f32
    %415 = llvm.load %130 : !llvm.ptr<5> -> f32
    %416 = llvm.getelementptr %28[18] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %417 = llvm.load %416 : !llvm.ptr<5> -> f32
    %418 = llvm.fmul %415, %417  : f32
    %419 = llvm.fadd %418, %414  : f32
    llvm.store %419, %31 : f32, !llvm.ptr<5>
    %420 = llvm.load %32 : !llvm.ptr<5> -> f32
    %421 = llvm.load %130 : !llvm.ptr<5> -> f32
    %422 = llvm.getelementptr %28[19] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %423 = llvm.load %422 : !llvm.ptr<5> -> f32
    %424 = llvm.fmul %421, %423  : f32
    %425 = llvm.fadd %424, %420  : f32
    llvm.store %425, %32 : f32, !llvm.ptr<5>
    %426 = llvm.load %33 : !llvm.ptr<5> -> f32
    %427 = llvm.getelementptr %28[9] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %428 = llvm.load %427 : !llvm.ptr<5> -> f32
    %429 = llvm.load %134 : !llvm.ptr<5> -> f32
    %430 = llvm.fmul %428, %429  : f32
    %431 = llvm.fadd %430, %426  : f32
    llvm.store %431, %33 : f32, !llvm.ptr<5>
    %432 = llvm.load %34 : !llvm.ptr<5> -> f32
    %433 = llvm.load %427 : !llvm.ptr<5> -> f32
    %434 = llvm.load %410 : !llvm.ptr<5> -> f32
    %435 = llvm.fmul %433, %434  : f32
    %436 = llvm.fadd %435, %432  : f32
    llvm.store %436, %34 : f32, !llvm.ptr<5>
    %437 = llvm.load %35 : !llvm.ptr<5> -> f32
    %438 = llvm.load %427 : !llvm.ptr<5> -> f32
    %439 = llvm.load %416 : !llvm.ptr<5> -> f32
    %440 = llvm.fmul %438, %439  : f32
    %441 = llvm.fadd %440, %437  : f32
    llvm.store %441, %35 : f32, !llvm.ptr<5>
    %442 = llvm.load %36 : !llvm.ptr<5> -> f32
    %443 = llvm.load %427 : !llvm.ptr<5> -> f32
    %444 = llvm.load %422 : !llvm.ptr<5> -> f32
    %445 = llvm.fmul %443, %444  : f32
    %446 = llvm.fadd %445, %442  : f32
    llvm.store %446, %36 : f32, !llvm.ptr<5>
    %447 = llvm.load %37 : !llvm.ptr<5> -> f32
    %448 = llvm.getelementptr %28[10] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %449 = llvm.load %448 : !llvm.ptr<5> -> f32
    %450 = llvm.load %134 : !llvm.ptr<5> -> f32
    %451 = llvm.fmul %449, %450  : f32
    %452 = llvm.fadd %451, %447  : f32
    llvm.store %452, %37 : f32, !llvm.ptr<5>
    %453 = llvm.load %38 : !llvm.ptr<5> -> f32
    %454 = llvm.load %448 : !llvm.ptr<5> -> f32
    %455 = llvm.load %410 : !llvm.ptr<5> -> f32
    %456 = llvm.fmul %454, %455  : f32
    %457 = llvm.fadd %456, %453  : f32
    llvm.store %457, %38 : f32, !llvm.ptr<5>
    %458 = llvm.load %39 : !llvm.ptr<5> -> f32
    %459 = llvm.load %448 : !llvm.ptr<5> -> f32
    %460 = llvm.load %416 : !llvm.ptr<5> -> f32
    %461 = llvm.fmul %459, %460  : f32
    %462 = llvm.fadd %461, %458  : f32
    llvm.store %462, %39 : f32, !llvm.ptr<5>
    %463 = llvm.load %40 : !llvm.ptr<5> -> f32
    %464 = llvm.load %448 : !llvm.ptr<5> -> f32
    %465 = llvm.load %422 : !llvm.ptr<5> -> f32
    %466 = llvm.fmul %464, %465  : f32
    %467 = llvm.fadd %466, %463  : f32
    llvm.store %467, %40 : f32, !llvm.ptr<5>
    %468 = llvm.load %41 : !llvm.ptr<5> -> f32
    %469 = llvm.getelementptr %28[11] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %470 = llvm.load %469 : !llvm.ptr<5> -> f32
    %471 = llvm.load %134 : !llvm.ptr<5> -> f32
    %472 = llvm.fmul %470, %471  : f32
    %473 = llvm.fadd %472, %468  : f32
    llvm.store %473, %41 : f32, !llvm.ptr<5>
    %474 = llvm.load %42 : !llvm.ptr<5> -> f32
    %475 = llvm.load %469 : !llvm.ptr<5> -> f32
    %476 = llvm.load %410 : !llvm.ptr<5> -> f32
    %477 = llvm.fmul %475, %476  : f32
    %478 = llvm.fadd %477, %474  : f32
    llvm.store %478, %42 : f32, !llvm.ptr<5>
    %479 = llvm.load %43 : !llvm.ptr<5> -> f32
    %480 = llvm.load %469 : !llvm.ptr<5> -> f32
    %481 = llvm.load %416 : !llvm.ptr<5> -> f32
    %482 = llvm.fmul %480, %481  : f32
    %483 = llvm.fadd %482, %479  : f32
    llvm.store %483, %43 : f32, !llvm.ptr<5>
    %484 = llvm.load %44 : !llvm.ptr<5> -> f32
    %485 = llvm.load %469 : !llvm.ptr<5> -> f32
    %486 = llvm.load %422 : !llvm.ptr<5> -> f32
    %487 = llvm.fmul %485, %486  : f32
    %488 = llvm.fadd %487, %484  : f32
    llvm.store %488, %44 : f32, !llvm.ptr<5>
    llvm.cond_br %138, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %489 = llvm.load %28 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<1xf32>
    %490 = llvm.add %154, %20  : i32
    %491 = llvm.srem %490, %2  : i32
    %492 = llvm.icmp "slt" %491, %22 : i32
    %493 = llvm.add %491, %2  : i32
    %494 = llvm.select %492, %493, %491 : i1, i32
    %495 = llvm.mul %494, %21  : i32
    %496 = llvm.add %495, %81  : i32
    %497 = llvm.add %496, %52  : i32
    %498 = llvm.getelementptr %27[%497] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.store %489, %498 {alignment = 4 : i64} : vector<1xf32>, !llvm.ptr<3>
    %499 = llvm.load %84 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<1xf32>
    %500 = llvm.add %495, %87  : i32
    %501 = llvm.add %500, %52  : i32
    %502 = llvm.getelementptr %27[%501] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.store %499, %502 {alignment = 4 : i64} : vector<1xf32>, !llvm.ptr<3>
    %503 = llvm.load %90 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<1xf32>
    %504 = llvm.add %495, %93  : i32
    %505 = llvm.add %504, %52  : i32
    %506 = llvm.getelementptr %27[%505] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.store %503, %506 {alignment = 4 : i64} : vector<1xf32>, !llvm.ptr<3>
    %507 = llvm.load %96 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<1xf32>
    %508 = llvm.add %495, %99  : i32
    %509 = llvm.add %508, %52  : i32
    %510 = llvm.getelementptr %27[%509] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.store %507, %510 {alignment = 4 : i64} : vector<1xf32>, !llvm.ptr<3>
    %511 = llvm.load %79 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<4xf32>
    %512 = llvm.add %495, %103  : i32
    %513 = llvm.add %512, %72  : i32
    %514 = llvm.add %513, %15  : i32
    %515 = llvm.getelementptr %27[%514] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    llvm.store %511, %515 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<3>
    rocdl.barrier
    llvm.br ^bb9
  ^bb9:  // 2 preds: ^bb7, ^bb8
    %516 = llvm.load %29 : !llvm.ptr<5> -> f32
    %517 = llvm.load %397 : !llvm.ptr<5> -> f32
    %518 = llvm.load %402 : !llvm.ptr<5> -> f32
    %519 = llvm.fmul %517, %518  : f32
    %520 = llvm.fadd %519, %516  : f32
    llvm.store %520, %29 : f32, !llvm.ptr<5>
    %521 = llvm.load %30 : !llvm.ptr<5> -> f32
    %522 = llvm.load %397 : !llvm.ptr<5> -> f32
    %523 = llvm.getelementptr %28[21] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %524 = llvm.load %523 : !llvm.ptr<5> -> f32
    %525 = llvm.fmul %522, %524  : f32
    %526 = llvm.fadd %525, %521  : f32
    llvm.store %526, %30 : f32, !llvm.ptr<5>
    %527 = llvm.load %31 : !llvm.ptr<5> -> f32
    %528 = llvm.load %397 : !llvm.ptr<5> -> f32
    %529 = llvm.getelementptr %28[22] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %530 = llvm.load %529 : !llvm.ptr<5> -> f32
    %531 = llvm.fmul %528, %530  : f32
    %532 = llvm.fadd %531, %527  : f32
    llvm.store %532, %31 : f32, !llvm.ptr<5>
    %533 = llvm.load %32 : !llvm.ptr<5> -> f32
    %534 = llvm.load %397 : !llvm.ptr<5> -> f32
    %535 = llvm.getelementptr %28[23] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %536 = llvm.load %535 : !llvm.ptr<5> -> f32
    %537 = llvm.fmul %534, %536  : f32
    %538 = llvm.fadd %537, %533  : f32
    llvm.store %538, %32 : f32, !llvm.ptr<5>
    %539 = llvm.load %33 : !llvm.ptr<5> -> f32
    %540 = llvm.getelementptr %28[13] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %541 = llvm.load %540 : !llvm.ptr<5> -> f32
    %542 = llvm.load %402 : !llvm.ptr<5> -> f32
    %543 = llvm.fmul %541, %542  : f32
    %544 = llvm.fadd %543, %539  : f32
    llvm.store %544, %33 : f32, !llvm.ptr<5>
    %545 = llvm.load %34 : !llvm.ptr<5> -> f32
    %546 = llvm.load %540 : !llvm.ptr<5> -> f32
    %547 = llvm.load %523 : !llvm.ptr<5> -> f32
    %548 = llvm.fmul %546, %547  : f32
    %549 = llvm.fadd %548, %545  : f32
    llvm.store %549, %34 : f32, !llvm.ptr<5>
    %550 = llvm.load %35 : !llvm.ptr<5> -> f32
    %551 = llvm.load %540 : !llvm.ptr<5> -> f32
    %552 = llvm.load %529 : !llvm.ptr<5> -> f32
    %553 = llvm.fmul %551, %552  : f32
    %554 = llvm.fadd %553, %550  : f32
    llvm.store %554, %35 : f32, !llvm.ptr<5>
    %555 = llvm.load %36 : !llvm.ptr<5> -> f32
    %556 = llvm.load %540 : !llvm.ptr<5> -> f32
    %557 = llvm.load %535 : !llvm.ptr<5> -> f32
    %558 = llvm.fmul %556, %557  : f32
    %559 = llvm.fadd %558, %555  : f32
    llvm.store %559, %36 : f32, !llvm.ptr<5>
    %560 = llvm.load %37 : !llvm.ptr<5> -> f32
    %561 = llvm.getelementptr %28[14] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %562 = llvm.load %561 : !llvm.ptr<5> -> f32
    %563 = llvm.load %402 : !llvm.ptr<5> -> f32
    %564 = llvm.fmul %562, %563  : f32
    %565 = llvm.fadd %564, %560  : f32
    llvm.store %565, %37 : f32, !llvm.ptr<5>
    %566 = llvm.load %38 : !llvm.ptr<5> -> f32
    %567 = llvm.load %561 : !llvm.ptr<5> -> f32
    %568 = llvm.load %523 : !llvm.ptr<5> -> f32
    %569 = llvm.fmul %567, %568  : f32
    %570 = llvm.fadd %569, %566  : f32
    llvm.store %570, %38 : f32, !llvm.ptr<5>
    %571 = llvm.load %39 : !llvm.ptr<5> -> f32
    %572 = llvm.load %561 : !llvm.ptr<5> -> f32
    %573 = llvm.load %529 : !llvm.ptr<5> -> f32
    %574 = llvm.fmul %572, %573  : f32
    %575 = llvm.fadd %574, %571  : f32
    llvm.store %575, %39 : f32, !llvm.ptr<5>
    %576 = llvm.load %40 : !llvm.ptr<5> -> f32
    %577 = llvm.load %561 : !llvm.ptr<5> -> f32
    %578 = llvm.load %535 : !llvm.ptr<5> -> f32
    %579 = llvm.fmul %577, %578  : f32
    %580 = llvm.fadd %579, %576  : f32
    llvm.store %580, %40 : f32, !llvm.ptr<5>
    %581 = llvm.load %41 : !llvm.ptr<5> -> f32
    %582 = llvm.getelementptr %28[15] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
    %583 = llvm.load %582 : !llvm.ptr<5> -> f32
    %584 = llvm.load %402 : !llvm.ptr<5> -> f32
    %585 = llvm.fmul %583, %584  : f32
    %586 = llvm.fadd %585, %581  : f32
    llvm.store %586, %41 : f32, !llvm.ptr<5>
    %587 = llvm.load %42 : !llvm.ptr<5> -> f32
    %588 = llvm.load %582 : !llvm.ptr<5> -> f32
    %589 = llvm.load %523 : !llvm.ptr<5> -> f32
    %590 = llvm.fmul %588, %589  : f32
    %591 = llvm.fadd %590, %587  : f32
    llvm.store %591, %42 : f32, !llvm.ptr<5>
    %592 = llvm.load %43 : !llvm.ptr<5> -> f32
    %593 = llvm.load %582 : !llvm.ptr<5> -> f32
    %594 = llvm.load %529 : !llvm.ptr<5> -> f32
    %595 = llvm.fmul %593, %594  : f32
    %596 = llvm.fadd %595, %592  : f32
    llvm.store %596, %43 : f32, !llvm.ptr<5>
    %597 = llvm.load %44 : !llvm.ptr<5> -> f32
    %598 = llvm.load %582 : !llvm.ptr<5> -> f32
    %599 = llvm.load %535 : !llvm.ptr<5> -> f32
    %600 = llvm.fmul %598, %599  : f32
    %601 = llvm.fadd %600, %597  : f32
    llvm.store %601, %44 : f32, !llvm.ptr<5>
    %602 = llvm.add %154, %20  : i32
    %603 = llvm.srem %602, %2  : i32
    %604 = llvm.icmp "slt" %603, %22 : i32
    %605 = llvm.add %603, %2  : i32
    %606 = llvm.select %604, %605, %603 : i1, i32
    %607 = llvm.mul %606, %21  : i32
    %608 = llvm.add %607, %72  : i32
    %609 = llvm.add %608, %15  : i32
    %610 = llvm.getelementptr %27[%609] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %611 = llvm.load %610 {alignment = 4 : i64} : !llvm.ptr<3> -> vector<4xf32>
    llvm.store %611, %134 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    %612 = llvm.add %607, %127  : i32
    %613 = llvm.getelementptr %27[%612] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    %614 = llvm.load %613 {alignment = 4 : i64} : !llvm.ptr<3> -> vector<4xf32>
    llvm.store %614, %130 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<5>
    %615 = llvm.add %135, %11  : i32
    llvm.br ^bb1(%615 : i32)
  ^bb10:  // pred: ^bb1
    %616 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<4xf32>
    %617 = llvm.add %53, %127  : i32
    %618 = llvm.add %73, %72  : i32
    %619 = llvm.mul %617, %21  : i32
    %620 = llvm.add %619, %618  : i32
    %621 = llvm.getelementptr %arg2[%620] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    llvm.store %616, %621 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<1>
    %622 = llvm.load %33 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<4xf32>
    %623 = llvm.add %617, %20  : i32
    %624 = llvm.mul %623, %21  : i32
    %625 = llvm.add %624, %618  : i32
    %626 = llvm.getelementptr %arg2[%625] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    llvm.store %622, %626 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<1>
    %627 = llvm.load %37 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<4xf32>
    %628 = llvm.add %617, %2  : i32
    %629 = llvm.mul %628, %21  : i32
    %630 = llvm.add %629, %618  : i32
    %631 = llvm.getelementptr %arg2[%630] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    llvm.store %627, %631 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<1>
    %632 = llvm.load %41 {alignment = 4 : i64} : !llvm.ptr<5> -> vector<4xf32>
    %633 = llvm.add %617, %0  : i32
    %634 = llvm.mul %633, %21  : i32
    %635 = llvm.add %634, %618  : i32
    %636 = llvm.getelementptr %arg2[%635] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    llvm.store %632, %636 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr<1>
    llvm.return
  }
}