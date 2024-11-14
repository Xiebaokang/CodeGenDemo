; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@global_smem = external addrspace(3) global [0 x i8]

declare ptr @malloc(i64 %0)

declare void @free(ptr %0)

define void @matmul_kernel_0d1d2d3de4de5de6de7c8de9c10de11c(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8) !dbg !4 {
  %10 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %11 = urem i32 %10, 64, !dbg !7
  %12 = udiv i32 %10, 64, !dbg !7
  %13 = udiv i32 %12, 1, !dbg !7
  %14 = urem i32 %13, 8, !dbg !7
  %15 = udiv i32 %11, 4, !dbg !7
  %16 = urem i32 %15, 16, !dbg !7
  %17 = urem i32 %14, 8, !dbg !7
  %18 = urem i32 %16, 128, !dbg !7
  %19 = mul i32 %17, 16, !dbg !7
  %20 = add i32 %18, %19, !dbg !7
  %21 = mul i32 %20, 1, !dbg !7
  %22 = add i32 %21, 0, !dbg !7
  %23 = add i32 %22, 0, !dbg !7
  %24 = urem i32 %12, 1, !dbg !7
  %25 = urem i32 %11, 64, !dbg !7
  %26 = urem i32 %24, 1, !dbg !7
  %27 = urem i32 %25, 64, !dbg !7
  %28 = mul i32 %26, 64, !dbg !7
  %29 = add i32 %27, %28, !dbg !7
  %30 = mul i32 %29, 2, !dbg !7
  %31 = add i32 %30, 0, !dbg !7
  %32 = add i32 %31, 0, !dbg !7
  %33 = add i32 %31, 1, !dbg !7
  %34 = udiv i32 %11, 32, !dbg !7
  %35 = urem i32 %34, 2, !dbg !7
  %36 = urem i32 %14, 64, !dbg !7
  %37 = urem i32 %35, 128, !dbg !7
  %38 = mul i32 %36, 2, !dbg !7
  %39 = add i32 %37, %38, !dbg !7
  %40 = mul i32 %39, 1, !dbg !7
  %41 = add i32 %40, 0, !dbg !7
  %42 = add i32 %41, 0, !dbg !7
  %43 = add i32 %41, 16, !dbg !7
  %44 = add i32 %41, 32, !dbg !7
  %45 = add i32 %41, 48, !dbg !7
  %46 = add i32 %41, 64, !dbg !7
  %47 = add i32 %41, 80, !dbg !7
  %48 = add i32 %41, 96, !dbg !7
  %49 = add i32 %41, 112, !dbg !7
  %50 = urem i32 %11, 32, !dbg !7
  %51 = urem i32 %50, 32, !dbg !7
  %52 = mul i32 %26, 32, !dbg !7
  %53 = add i32 %51, %52, !dbg !7
  %54 = mul i32 %53, 4, !dbg !7
  %55 = add i32 %54, 0, !dbg !7
  %56 = add i32 %55, 0, !dbg !7
  %57 = add i32 %55, 1, !dbg !7
  %58 = add i32 %55, 2, !dbg !7
  %59 = add i32 %55, 3, !dbg !7
  %60 = urem i32 %11, 4, !dbg !8
  %61 = urem i32 %60, 4, !dbg !8
  %62 = mul i32 %26, 4, !dbg !8
  %63 = add i32 %61, %62, !dbg !8
  %64 = mul i32 %63, 2, !dbg !8
  %65 = add i32 %64, 0, !dbg !8
  %66 = add i32 %65, 0, !dbg !8
  %67 = add i32 %65, 1, !dbg !8
  %68 = udiv i32 %11, 64, !dbg !9
  %69 = urem i32 %68, 1, !dbg !9
  %70 = urem i32 %69, 8, !dbg !9
  %71 = mul i32 %17, 1, !dbg !9
  %72 = add i32 %70, %71, !dbg !9
  %73 = mul i32 %72, 1, !dbg !9
  %74 = add i32 %73, 0, !dbg !9
  %75 = add i32 %74, 0, !dbg !9
  %76 = add i32 %21, 0, !dbg !10
  %77 = add i32 %64, 0, !dbg !10
  %78 = add i32 %73, 0, !dbg !11
  %79 = add i32 %30, 0, !dbg !11
  %80 = urem i32 %14, 16, !dbg !12
  %81 = urem i32 %35, 32, !dbg !12
  %82 = mul i32 %80, 2, !dbg !12
  %83 = add i32 %81, %82, !dbg !12
  %84 = mul i32 %83, 4, !dbg !12
  %85 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !13
  %86 = add i32 %3, 127, !dbg !14
  %87 = sdiv i32 %86, 128, !dbg !18
  %88 = add i32 %4, 127, !dbg !19
  %89 = sdiv i32 %88, 128, !dbg !21
  %90 = mul i32 %89, 4, !dbg !22
  %91 = sdiv i32 %85, %90, !dbg !23
  %92 = mul i32 %91, 4, !dbg !24
  %93 = sub i32 %87, %92, !dbg !25
  %94 = call i32 @llvm.smin.i32(i32 %93, i32 4), !dbg !26
  %95 = srem i32 %85, %94, !dbg !28
  %96 = add i32 %92, %95, !dbg !29
  %97 = srem i32 %85, %90, !dbg !30
  %98 = sdiv i32 %97, %94, !dbg !31
  %99 = mul i32 %96, 128, !dbg !32
  %100 = add i32 %23, 0, !dbg !7
  %101 = add i32 %32, 0, !dbg !7
  %102 = add i32 %33, 0, !dbg !7
  %103 = add i32 %42, 0, !dbg !7
  %104 = add i32 %43, 0, !dbg !7
  %105 = add i32 %44, 0, !dbg !7
  %106 = add i32 %45, 0, !dbg !7
  %107 = add i32 %46, 0, !dbg !7
  %108 = add i32 %47, 0, !dbg !7
  %109 = add i32 %48, 0, !dbg !7
  %110 = add i32 %49, 0, !dbg !7
  %111 = add i32 %56, 0, !dbg !7
  %112 = add i32 %57, 0, !dbg !7
  %113 = add i32 %58, 0, !dbg !7
  %114 = add i32 %59, 0, !dbg !7
  %115 = add i32 %99, %100, !dbg !33
  %116 = add i32 %99, %103, !dbg !33
  %117 = add i32 %99, %104, !dbg !33
  %118 = add i32 %99, %105, !dbg !33
  %119 = add i32 %99, %106, !dbg !33
  %120 = add i32 %99, %107, !dbg !33
  %121 = add i32 %99, %108, !dbg !33
  %122 = add i32 %99, %109, !dbg !33
  %123 = add i32 %99, %110, !dbg !33
  %124 = srem i32 %115, %3, !dbg !34
  %125 = mul i32 %98, 128, !dbg !35
  %126 = add i32 %125, %101, !dbg !36
  %127 = add i32 %125, %102, !dbg !36
  %128 = add i32 %125, %111, !dbg !36
  %129 = add i32 %125, %112, !dbg !36
  %130 = add i32 %125, %113, !dbg !36
  %131 = add i32 %125, %114, !dbg !36
  %132 = srem i32 %126, %4, !dbg !37
  %133 = srem i32 %127, %4, !dbg !37
  %134 = mul i32 %124, %6, !dbg !38
  %135 = add i32 %66, 0, !dbg !8
  %136 = add i32 %67, 0, !dbg !8
  %137 = add i32 %134, %135, !dbg !8
  %138 = add i32 %134, %136, !dbg !8
  %139 = getelementptr float, ptr addrspace(1) %0, i32 %137, !dbg !39
  %140 = getelementptr float, ptr addrspace(1) %0, i32 %138, !dbg !39
  %141 = insertvalue { ptr addrspace(1), ptr addrspace(1) } undef, ptr addrspace(1) %139, 0, !dbg !39
  %142 = insertvalue { ptr addrspace(1), ptr addrspace(1) } %141, ptr addrspace(1) %140, 1, !dbg !39
  %143 = add i32 %75, 0, !dbg !9
  %144 = mul i32 %143, %7, !dbg !40
  %145 = add i32 %144, %132, !dbg !41
  %146 = add i32 %144, %133, !dbg !41
  %147 = getelementptr float, ptr addrspace(1) %1, i32 %145, !dbg !9
  %148 = getelementptr float, ptr addrspace(1) %1, i32 %146, !dbg !9
  %149 = insertvalue { ptr addrspace(1), ptr addrspace(1) } undef, ptr addrspace(1) %147, 0, !dbg !9
  %150 = insertvalue { ptr addrspace(1), ptr addrspace(1) } %149, ptr addrspace(1) %148, 1, !dbg !9
  %151 = add i32 %5, 7, !dbg !42
  %152 = sdiv i32 %151, 8, !dbg !44
  %153 = mul i32 %7, 8, !dbg !45
  br label %154, !dbg !46

154:                                              ; preds = %160, %9
  %155 = phi i32 [ %740, %160 ], [ 0, %9 ]
  %156 = phi { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } [ %729, %160 ], [ zeroinitializer, %9 ]
  %157 = phi { ptr addrspace(1), ptr addrspace(1) } [ %734, %160 ], [ %142, %9 ]
  %158 = phi { ptr addrspace(1), ptr addrspace(1) } [ %739, %160 ], [ %150, %9 ]
  %159 = icmp slt i32 %155, %152, !dbg !46
  br i1 %159, label %160, label %741, !dbg !46

160:                                              ; preds = %154
  %161 = extractvalue { ptr addrspace(1), ptr addrspace(1) } %157, 0, !dbg !10
  %162 = addrspacecast ptr addrspace(1) %161 to ptr, !dbg !10
  %163 = load <2 x float>, ptr %162, align 8, !dbg !10
  %164 = extractelement <2 x float> %163, i32 0, !dbg !10
  %165 = extractelement <2 x float> %163, i32 1, !dbg !10
  %166 = extractvalue { ptr addrspace(1), ptr addrspace(1) } %158, 0, !dbg !11
  %167 = addrspacecast ptr addrspace(1) %166 to ptr, !dbg !11
  %168 = load <2 x float>, ptr %167, align 8, !dbg !11
  %169 = extractelement <2 x float> %168, i32 0, !dbg !11
  %170 = extractelement <2 x float> %168, i32 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !10
  call void @llvm.amdgcn.s.barrier(), !dbg !10
  fence syncscope("workgroup") acquire, !dbg !10
  %171 = udiv i32 %76, 1, !dbg !10
  %172 = urem i32 %171, 1, !dbg !10
  %173 = mul i32 %76, 8, !dbg !10
  %174 = udiv i32 %77, 1, !dbg !10
  %175 = xor i32 %174, %172, !dbg !10
  %176 = mul i32 %175, 1, !dbg !10
  %177 = urem i32 %77, 1, !dbg !10
  %178 = udiv i32 %177, 1, !dbg !10
  %179 = mul i32 %178, 1, !dbg !10
  %180 = add i32 %176, %179, !dbg !10
  %181 = mul i32 %180, 1, !dbg !10
  %182 = add i32 %173, %181, !dbg !10
  %183 = add i32 %182, 0, !dbg !10
  %184 = getelementptr float, ptr addrspace(3) @global_smem, i32 %183, !dbg !10
  %185 = getelementptr float, ptr addrspace(3) %184, i32 0, !dbg !10
  %186 = getelementptr float, ptr addrspace(3) %184, i32 1, !dbg !10
  %187 = insertelement <1 x float> undef, float %164, i32 0, !dbg !10
  store <1 x float> %187, ptr addrspace(3) %185, align 4, !dbg !10
  %188 = insertelement <1 x float> undef, float %165, i32 0, !dbg !10
  store <1 x float> %188, ptr addrspace(3) %186, align 4, !dbg !10
  fence syncscope("workgroup") release, !dbg !10
  call void @llvm.amdgcn.s.barrier(), !dbg !10
  fence syncscope("workgroup") acquire, !dbg !10
  %189 = udiv i32 %10, 32, !dbg !10
  %190 = urem i32 %189, 64, !dbg !10
  %191 = mul i32 %190, 4, !dbg !10
  %192 = mul i32 %191, 8, !dbg !10
  %193 = add i32 0, %192, !dbg !10
  %194 = getelementptr float, ptr addrspace(3) @global_smem, i32 %193, !dbg !10
  %195 = getelementptr float, ptr addrspace(3) %194, i32 0, !dbg !10
  %196 = load float, ptr addrspace(3) %195, align 4, !dbg !10
  %197 = getelementptr float, ptr addrspace(3) %194, i32 8, !dbg !10
  %198 = load float, ptr addrspace(3) %197, align 4, !dbg !10
  %199 = getelementptr float, ptr addrspace(3) %194, i32 16, !dbg !10
  %200 = load float, ptr addrspace(3) %199, align 4, !dbg !10
  %201 = getelementptr float, ptr addrspace(3) %194, i32 24, !dbg !10
  %202 = load float, ptr addrspace(3) %201, align 4, !dbg !10
  %203 = getelementptr float, ptr addrspace(3) %194, i32 512, !dbg !10
  %204 = load float, ptr addrspace(3) %203, align 4, !dbg !10
  %205 = getelementptr float, ptr addrspace(3) %194, i32 520, !dbg !10
  %206 = load float, ptr addrspace(3) %205, align 4, !dbg !10
  %207 = getelementptr float, ptr addrspace(3) %194, i32 528, !dbg !10
  %208 = load float, ptr addrspace(3) %207, align 4, !dbg !10
  %209 = getelementptr float, ptr addrspace(3) %194, i32 536, !dbg !10
  %210 = load float, ptr addrspace(3) %209, align 4, !dbg !10
  %211 = getelementptr float, ptr addrspace(3) %194, i32 1, !dbg !10
  %212 = load float, ptr addrspace(3) %211, align 4, !dbg !10
  %213 = getelementptr float, ptr addrspace(3) %194, i32 9, !dbg !10
  %214 = load float, ptr addrspace(3) %213, align 4, !dbg !10
  %215 = getelementptr float, ptr addrspace(3) %194, i32 17, !dbg !10
  %216 = load float, ptr addrspace(3) %215, align 4, !dbg !10
  %217 = getelementptr float, ptr addrspace(3) %194, i32 25, !dbg !10
  %218 = load float, ptr addrspace(3) %217, align 4, !dbg !10
  %219 = getelementptr float, ptr addrspace(3) %194, i32 513, !dbg !10
  %220 = load float, ptr addrspace(3) %219, align 4, !dbg !10
  %221 = getelementptr float, ptr addrspace(3) %194, i32 521, !dbg !10
  %222 = load float, ptr addrspace(3) %221, align 4, !dbg !10
  %223 = getelementptr float, ptr addrspace(3) %194, i32 529, !dbg !10
  %224 = load float, ptr addrspace(3) %223, align 4, !dbg !10
  %225 = getelementptr float, ptr addrspace(3) %194, i32 537, !dbg !10
  %226 = load float, ptr addrspace(3) %225, align 4, !dbg !10
  %227 = getelementptr float, ptr addrspace(3) %194, i32 2, !dbg !10
  %228 = load float, ptr addrspace(3) %227, align 4, !dbg !10
  %229 = getelementptr float, ptr addrspace(3) %194, i32 10, !dbg !10
  %230 = load float, ptr addrspace(3) %229, align 4, !dbg !10
  %231 = getelementptr float, ptr addrspace(3) %194, i32 18, !dbg !10
  %232 = load float, ptr addrspace(3) %231, align 4, !dbg !10
  %233 = getelementptr float, ptr addrspace(3) %194, i32 26, !dbg !10
  %234 = load float, ptr addrspace(3) %233, align 4, !dbg !10
  %235 = getelementptr float, ptr addrspace(3) %194, i32 514, !dbg !10
  %236 = load float, ptr addrspace(3) %235, align 4, !dbg !10
  %237 = getelementptr float, ptr addrspace(3) %194, i32 522, !dbg !10
  %238 = load float, ptr addrspace(3) %237, align 4, !dbg !10
  %239 = getelementptr float, ptr addrspace(3) %194, i32 530, !dbg !10
  %240 = load float, ptr addrspace(3) %239, align 4, !dbg !10
  %241 = getelementptr float, ptr addrspace(3) %194, i32 538, !dbg !10
  %242 = load float, ptr addrspace(3) %241, align 4, !dbg !10
  %243 = getelementptr float, ptr addrspace(3) %194, i32 3, !dbg !10
  %244 = load float, ptr addrspace(3) %243, align 4, !dbg !10
  %245 = getelementptr float, ptr addrspace(3) %194, i32 11, !dbg !10
  %246 = load float, ptr addrspace(3) %245, align 4, !dbg !10
  %247 = getelementptr float, ptr addrspace(3) %194, i32 19, !dbg !10
  %248 = load float, ptr addrspace(3) %247, align 4, !dbg !10
  %249 = getelementptr float, ptr addrspace(3) %194, i32 27, !dbg !10
  %250 = load float, ptr addrspace(3) %249, align 4, !dbg !10
  %251 = getelementptr float, ptr addrspace(3) %194, i32 515, !dbg !10
  %252 = load float, ptr addrspace(3) %251, align 4, !dbg !10
  %253 = getelementptr float, ptr addrspace(3) %194, i32 523, !dbg !10
  %254 = load float, ptr addrspace(3) %253, align 4, !dbg !10
  %255 = getelementptr float, ptr addrspace(3) %194, i32 531, !dbg !10
  %256 = load float, ptr addrspace(3) %255, align 4, !dbg !10
  %257 = getelementptr float, ptr addrspace(3) %194, i32 539, !dbg !10
  %258 = load float, ptr addrspace(3) %257, align 4, !dbg !10
  %259 = getelementptr float, ptr addrspace(3) %194, i32 4, !dbg !10
  %260 = load float, ptr addrspace(3) %259, align 4, !dbg !10
  %261 = getelementptr float, ptr addrspace(3) %194, i32 12, !dbg !10
  %262 = load float, ptr addrspace(3) %261, align 4, !dbg !10
  %263 = getelementptr float, ptr addrspace(3) %194, i32 20, !dbg !10
  %264 = load float, ptr addrspace(3) %263, align 4, !dbg !10
  %265 = getelementptr float, ptr addrspace(3) %194, i32 28, !dbg !10
  %266 = load float, ptr addrspace(3) %265, align 4, !dbg !10
  %267 = getelementptr float, ptr addrspace(3) %194, i32 516, !dbg !10
  %268 = load float, ptr addrspace(3) %267, align 4, !dbg !10
  %269 = getelementptr float, ptr addrspace(3) %194, i32 524, !dbg !10
  %270 = load float, ptr addrspace(3) %269, align 4, !dbg !10
  %271 = getelementptr float, ptr addrspace(3) %194, i32 532, !dbg !10
  %272 = load float, ptr addrspace(3) %271, align 4, !dbg !10
  %273 = getelementptr float, ptr addrspace(3) %194, i32 540, !dbg !10
  %274 = load float, ptr addrspace(3) %273, align 4, !dbg !10
  %275 = getelementptr float, ptr addrspace(3) %194, i32 5, !dbg !10
  %276 = load float, ptr addrspace(3) %275, align 4, !dbg !10
  %277 = getelementptr float, ptr addrspace(3) %194, i32 13, !dbg !10
  %278 = load float, ptr addrspace(3) %277, align 4, !dbg !10
  %279 = getelementptr float, ptr addrspace(3) %194, i32 21, !dbg !10
  %280 = load float, ptr addrspace(3) %279, align 4, !dbg !10
  %281 = getelementptr float, ptr addrspace(3) %194, i32 29, !dbg !10
  %282 = load float, ptr addrspace(3) %281, align 4, !dbg !10
  %283 = getelementptr float, ptr addrspace(3) %194, i32 517, !dbg !10
  %284 = load float, ptr addrspace(3) %283, align 4, !dbg !10
  %285 = getelementptr float, ptr addrspace(3) %194, i32 525, !dbg !10
  %286 = load float, ptr addrspace(3) %285, align 4, !dbg !10
  %287 = getelementptr float, ptr addrspace(3) %194, i32 533, !dbg !10
  %288 = load float, ptr addrspace(3) %287, align 4, !dbg !10
  %289 = getelementptr float, ptr addrspace(3) %194, i32 541, !dbg !10
  %290 = load float, ptr addrspace(3) %289, align 4, !dbg !10
  %291 = getelementptr float, ptr addrspace(3) %194, i32 6, !dbg !10
  %292 = load float, ptr addrspace(3) %291, align 4, !dbg !10
  %293 = getelementptr float, ptr addrspace(3) %194, i32 14, !dbg !10
  %294 = load float, ptr addrspace(3) %293, align 4, !dbg !10
  %295 = getelementptr float, ptr addrspace(3) %194, i32 22, !dbg !10
  %296 = load float, ptr addrspace(3) %295, align 4, !dbg !10
  %297 = getelementptr float, ptr addrspace(3) %194, i32 30, !dbg !10
  %298 = load float, ptr addrspace(3) %297, align 4, !dbg !10
  %299 = getelementptr float, ptr addrspace(3) %194, i32 518, !dbg !10
  %300 = load float, ptr addrspace(3) %299, align 4, !dbg !10
  %301 = getelementptr float, ptr addrspace(3) %194, i32 526, !dbg !10
  %302 = load float, ptr addrspace(3) %301, align 4, !dbg !10
  %303 = getelementptr float, ptr addrspace(3) %194, i32 534, !dbg !10
  %304 = load float, ptr addrspace(3) %303, align 4, !dbg !10
  %305 = getelementptr float, ptr addrspace(3) %194, i32 542, !dbg !10
  %306 = load float, ptr addrspace(3) %305, align 4, !dbg !10
  %307 = getelementptr float, ptr addrspace(3) %194, i32 7, !dbg !10
  %308 = load float, ptr addrspace(3) %307, align 4, !dbg !10
  %309 = getelementptr float, ptr addrspace(3) %194, i32 15, !dbg !10
  %310 = load float, ptr addrspace(3) %309, align 4, !dbg !10
  %311 = getelementptr float, ptr addrspace(3) %194, i32 23, !dbg !10
  %312 = load float, ptr addrspace(3) %311, align 4, !dbg !10
  %313 = getelementptr float, ptr addrspace(3) %194, i32 31, !dbg !10
  %314 = load float, ptr addrspace(3) %313, align 4, !dbg !10
  %315 = getelementptr float, ptr addrspace(3) %194, i32 519, !dbg !10
  %316 = load float, ptr addrspace(3) %315, align 4, !dbg !10
  %317 = getelementptr float, ptr addrspace(3) %194, i32 527, !dbg !10
  %318 = load float, ptr addrspace(3) %317, align 4, !dbg !10
  %319 = getelementptr float, ptr addrspace(3) %194, i32 535, !dbg !10
  %320 = load float, ptr addrspace(3) %319, align 4, !dbg !10
  %321 = getelementptr float, ptr addrspace(3) %194, i32 543, !dbg !10
  %322 = load float, ptr addrspace(3) %321, align 4, !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %323 = udiv i32 %78, 1, !dbg !11
  %324 = urem i32 %323, 1, !dbg !11
  %325 = mul i32 %78, 128, !dbg !11
  %326 = udiv i32 %79, 1, !dbg !11
  %327 = xor i32 %326, %324, !dbg !11
  %328 = mul i32 %327, 1, !dbg !11
  %329 = urem i32 %79, 1, !dbg !11
  %330 = udiv i32 %329, 1, !dbg !11
  %331 = mul i32 %330, 1, !dbg !11
  %332 = add i32 %328, %331, !dbg !11
  %333 = mul i32 %332, 1, !dbg !11
  %334 = add i32 %325, %333, !dbg !11
  %335 = add i32 %334, 0, !dbg !11
  %336 = getelementptr float, ptr addrspace(3) @global_smem, i32 %335, !dbg !11
  %337 = getelementptr float, ptr addrspace(3) %336, i32 0, !dbg !11
  %338 = getelementptr float, ptr addrspace(3) %336, i32 1, !dbg !11
  %339 = insertelement <1 x float> undef, float %169, i32 0, !dbg !11
  store <1 x float> %339, ptr addrspace(3) %337, align 4, !dbg !11
  %340 = insertelement <1 x float> undef, float %170, i32 0, !dbg !11
  store <1 x float> %340, ptr addrspace(3) %338, align 4, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %341 = urem i32 %10, 32, !dbg !11
  %342 = mul i32 %341, 4, !dbg !11
  %343 = mul i32 %342, 1, !dbg !11
  %344 = add i32 %343, 0, !dbg !11
  %345 = getelementptr float, ptr addrspace(3) @global_smem, i32 %344, !dbg !11
  %346 = getelementptr float, ptr addrspace(3) %345, i32 0, !dbg !11
  %347 = load float, ptr addrspace(3) %346, align 4, !dbg !11
  %348 = getelementptr float, ptr addrspace(3) %345, i32 1, !dbg !11
  %349 = load float, ptr addrspace(3) %348, align 4, !dbg !11
  %350 = getelementptr float, ptr addrspace(3) %345, i32 2, !dbg !11
  %351 = load float, ptr addrspace(3) %350, align 4, !dbg !11
  %352 = getelementptr float, ptr addrspace(3) %345, i32 3, !dbg !11
  %353 = load float, ptr addrspace(3) %352, align 4, !dbg !11
  %354 = getelementptr float, ptr addrspace(3) %345, i32 128, !dbg !11
  %355 = load float, ptr addrspace(3) %354, align 4, !dbg !11
  %356 = getelementptr float, ptr addrspace(3) %345, i32 129, !dbg !11
  %357 = load float, ptr addrspace(3) %356, align 4, !dbg !11
  %358 = getelementptr float, ptr addrspace(3) %345, i32 130, !dbg !11
  %359 = load float, ptr addrspace(3) %358, align 4, !dbg !11
  %360 = getelementptr float, ptr addrspace(3) %345, i32 131, !dbg !11
  %361 = load float, ptr addrspace(3) %360, align 4, !dbg !11
  %362 = getelementptr float, ptr addrspace(3) %345, i32 256, !dbg !11
  %363 = load float, ptr addrspace(3) %362, align 4, !dbg !11
  %364 = getelementptr float, ptr addrspace(3) %345, i32 257, !dbg !11
  %365 = load float, ptr addrspace(3) %364, align 4, !dbg !11
  %366 = getelementptr float, ptr addrspace(3) %345, i32 258, !dbg !11
  %367 = load float, ptr addrspace(3) %366, align 4, !dbg !11
  %368 = getelementptr float, ptr addrspace(3) %345, i32 259, !dbg !11
  %369 = load float, ptr addrspace(3) %368, align 4, !dbg !11
  %370 = getelementptr float, ptr addrspace(3) %345, i32 384, !dbg !11
  %371 = load float, ptr addrspace(3) %370, align 4, !dbg !11
  %372 = getelementptr float, ptr addrspace(3) %345, i32 385, !dbg !11
  %373 = load float, ptr addrspace(3) %372, align 4, !dbg !11
  %374 = getelementptr float, ptr addrspace(3) %345, i32 386, !dbg !11
  %375 = load float, ptr addrspace(3) %374, align 4, !dbg !11
  %376 = getelementptr float, ptr addrspace(3) %345, i32 387, !dbg !11
  %377 = load float, ptr addrspace(3) %376, align 4, !dbg !11
  %378 = getelementptr float, ptr addrspace(3) %345, i32 512, !dbg !11
  %379 = load float, ptr addrspace(3) %378, align 4, !dbg !11
  %380 = getelementptr float, ptr addrspace(3) %345, i32 513, !dbg !11
  %381 = load float, ptr addrspace(3) %380, align 4, !dbg !11
  %382 = getelementptr float, ptr addrspace(3) %345, i32 514, !dbg !11
  %383 = load float, ptr addrspace(3) %382, align 4, !dbg !11
  %384 = getelementptr float, ptr addrspace(3) %345, i32 515, !dbg !11
  %385 = load float, ptr addrspace(3) %384, align 4, !dbg !11
  %386 = getelementptr float, ptr addrspace(3) %345, i32 640, !dbg !11
  %387 = load float, ptr addrspace(3) %386, align 4, !dbg !11
  %388 = getelementptr float, ptr addrspace(3) %345, i32 641, !dbg !11
  %389 = load float, ptr addrspace(3) %388, align 4, !dbg !11
  %390 = getelementptr float, ptr addrspace(3) %345, i32 642, !dbg !11
  %391 = load float, ptr addrspace(3) %390, align 4, !dbg !11
  %392 = getelementptr float, ptr addrspace(3) %345, i32 643, !dbg !11
  %393 = load float, ptr addrspace(3) %392, align 4, !dbg !11
  %394 = getelementptr float, ptr addrspace(3) %345, i32 768, !dbg !11
  %395 = load float, ptr addrspace(3) %394, align 4, !dbg !11
  %396 = getelementptr float, ptr addrspace(3) %345, i32 769, !dbg !11
  %397 = load float, ptr addrspace(3) %396, align 4, !dbg !11
  %398 = getelementptr float, ptr addrspace(3) %345, i32 770, !dbg !11
  %399 = load float, ptr addrspace(3) %398, align 4, !dbg !11
  %400 = getelementptr float, ptr addrspace(3) %345, i32 771, !dbg !11
  %401 = load float, ptr addrspace(3) %400, align 4, !dbg !11
  %402 = getelementptr float, ptr addrspace(3) %345, i32 896, !dbg !11
  %403 = load float, ptr addrspace(3) %402, align 4, !dbg !11
  %404 = getelementptr float, ptr addrspace(3) %345, i32 897, !dbg !11
  %405 = load float, ptr addrspace(3) %404, align 4, !dbg !11
  %406 = getelementptr float, ptr addrspace(3) %345, i32 898, !dbg !11
  %407 = load float, ptr addrspace(3) %406, align 4, !dbg !11
  %408 = getelementptr float, ptr addrspace(3) %345, i32 899, !dbg !11
  %409 = load float, ptr addrspace(3) %408, align 4, !dbg !11
  %410 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 0, !dbg !47
  %411 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 1, !dbg !47
  %412 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 2, !dbg !47
  %413 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 3, !dbg !47
  %414 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 4, !dbg !47
  %415 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 5, !dbg !47
  %416 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 6, !dbg !47
  %417 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 7, !dbg !47
  %418 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 8, !dbg !47
  %419 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 9, !dbg !47
  %420 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 10, !dbg !47
  %421 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 11, !dbg !47
  %422 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 12, !dbg !47
  %423 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 13, !dbg !47
  %424 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 14, !dbg !47
  %425 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 15, !dbg !47
  %426 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 16, !dbg !47
  %427 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 17, !dbg !47
  %428 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 18, !dbg !47
  %429 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 19, !dbg !47
  %430 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 20, !dbg !47
  %431 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 21, !dbg !47
  %432 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 22, !dbg !47
  %433 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 23, !dbg !47
  %434 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 24, !dbg !47
  %435 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 25, !dbg !47
  %436 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 26, !dbg !47
  %437 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 27, !dbg !47
  %438 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 28, !dbg !47
  %439 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 29, !dbg !47
  %440 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 30, !dbg !47
  %441 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 31, !dbg !47
  %442 = call float @llvm.fmuladd.f32(float %196, float %347, float %410), !dbg !47
  %443 = call float @llvm.fmuladd.f32(float %196, float %349, float %411), !dbg !47
  %444 = call float @llvm.fmuladd.f32(float %196, float %351, float %412), !dbg !47
  %445 = call float @llvm.fmuladd.f32(float %196, float %353, float %413), !dbg !47
  %446 = call float @llvm.fmuladd.f32(float %198, float %347, float %414), !dbg !47
  %447 = call float @llvm.fmuladd.f32(float %198, float %349, float %415), !dbg !47
  %448 = call float @llvm.fmuladd.f32(float %198, float %351, float %416), !dbg !47
  %449 = call float @llvm.fmuladd.f32(float %198, float %353, float %417), !dbg !47
  %450 = call float @llvm.fmuladd.f32(float %200, float %347, float %418), !dbg !47
  %451 = call float @llvm.fmuladd.f32(float %200, float %349, float %419), !dbg !47
  %452 = call float @llvm.fmuladd.f32(float %200, float %351, float %420), !dbg !47
  %453 = call float @llvm.fmuladd.f32(float %200, float %353, float %421), !dbg !47
  %454 = call float @llvm.fmuladd.f32(float %202, float %347, float %422), !dbg !47
  %455 = call float @llvm.fmuladd.f32(float %202, float %349, float %423), !dbg !47
  %456 = call float @llvm.fmuladd.f32(float %202, float %351, float %424), !dbg !47
  %457 = call float @llvm.fmuladd.f32(float %202, float %353, float %425), !dbg !47
  %458 = call float @llvm.fmuladd.f32(float %204, float %347, float %426), !dbg !47
  %459 = call float @llvm.fmuladd.f32(float %204, float %349, float %427), !dbg !47
  %460 = call float @llvm.fmuladd.f32(float %204, float %351, float %428), !dbg !47
  %461 = call float @llvm.fmuladd.f32(float %204, float %353, float %429), !dbg !47
  %462 = call float @llvm.fmuladd.f32(float %206, float %347, float %430), !dbg !47
  %463 = call float @llvm.fmuladd.f32(float %206, float %349, float %431), !dbg !47
  %464 = call float @llvm.fmuladd.f32(float %206, float %351, float %432), !dbg !47
  %465 = call float @llvm.fmuladd.f32(float %206, float %353, float %433), !dbg !47
  %466 = call float @llvm.fmuladd.f32(float %208, float %347, float %434), !dbg !47
  %467 = call float @llvm.fmuladd.f32(float %208, float %349, float %435), !dbg !47
  %468 = call float @llvm.fmuladd.f32(float %208, float %351, float %436), !dbg !47
  %469 = call float @llvm.fmuladd.f32(float %208, float %353, float %437), !dbg !47
  %470 = call float @llvm.fmuladd.f32(float %210, float %347, float %438), !dbg !47
  %471 = call float @llvm.fmuladd.f32(float %210, float %349, float %439), !dbg !47
  %472 = call float @llvm.fmuladd.f32(float %210, float %351, float %440), !dbg !47
  %473 = call float @llvm.fmuladd.f32(float %210, float %353, float %441), !dbg !47
  %474 = call float @llvm.fmuladd.f32(float %212, float %355, float %442), !dbg !47
  %475 = call float @llvm.fmuladd.f32(float %212, float %357, float %443), !dbg !47
  %476 = call float @llvm.fmuladd.f32(float %212, float %359, float %444), !dbg !47
  %477 = call float @llvm.fmuladd.f32(float %212, float %361, float %445), !dbg !47
  %478 = call float @llvm.fmuladd.f32(float %214, float %355, float %446), !dbg !47
  %479 = call float @llvm.fmuladd.f32(float %214, float %357, float %447), !dbg !47
  %480 = call float @llvm.fmuladd.f32(float %214, float %359, float %448), !dbg !47
  %481 = call float @llvm.fmuladd.f32(float %214, float %361, float %449), !dbg !47
  %482 = call float @llvm.fmuladd.f32(float %216, float %355, float %450), !dbg !47
  %483 = call float @llvm.fmuladd.f32(float %216, float %357, float %451), !dbg !47
  %484 = call float @llvm.fmuladd.f32(float %216, float %359, float %452), !dbg !47
  %485 = call float @llvm.fmuladd.f32(float %216, float %361, float %453), !dbg !47
  %486 = call float @llvm.fmuladd.f32(float %218, float %355, float %454), !dbg !47
  %487 = call float @llvm.fmuladd.f32(float %218, float %357, float %455), !dbg !47
  %488 = call float @llvm.fmuladd.f32(float %218, float %359, float %456), !dbg !47
  %489 = call float @llvm.fmuladd.f32(float %218, float %361, float %457), !dbg !47
  %490 = call float @llvm.fmuladd.f32(float %220, float %355, float %458), !dbg !47
  %491 = call float @llvm.fmuladd.f32(float %220, float %357, float %459), !dbg !47
  %492 = call float @llvm.fmuladd.f32(float %220, float %359, float %460), !dbg !47
  %493 = call float @llvm.fmuladd.f32(float %220, float %361, float %461), !dbg !47
  %494 = call float @llvm.fmuladd.f32(float %222, float %355, float %462), !dbg !47
  %495 = call float @llvm.fmuladd.f32(float %222, float %357, float %463), !dbg !47
  %496 = call float @llvm.fmuladd.f32(float %222, float %359, float %464), !dbg !47
  %497 = call float @llvm.fmuladd.f32(float %222, float %361, float %465), !dbg !47
  %498 = call float @llvm.fmuladd.f32(float %224, float %355, float %466), !dbg !47
  %499 = call float @llvm.fmuladd.f32(float %224, float %357, float %467), !dbg !47
  %500 = call float @llvm.fmuladd.f32(float %224, float %359, float %468), !dbg !47
  %501 = call float @llvm.fmuladd.f32(float %224, float %361, float %469), !dbg !47
  %502 = call float @llvm.fmuladd.f32(float %226, float %355, float %470), !dbg !47
  %503 = call float @llvm.fmuladd.f32(float %226, float %357, float %471), !dbg !47
  %504 = call float @llvm.fmuladd.f32(float %226, float %359, float %472), !dbg !47
  %505 = call float @llvm.fmuladd.f32(float %226, float %361, float %473), !dbg !47
  %506 = call float @llvm.fmuladd.f32(float %228, float %363, float %474), !dbg !47
  %507 = call float @llvm.fmuladd.f32(float %228, float %365, float %475), !dbg !47
  %508 = call float @llvm.fmuladd.f32(float %228, float %367, float %476), !dbg !47
  %509 = call float @llvm.fmuladd.f32(float %228, float %369, float %477), !dbg !47
  %510 = call float @llvm.fmuladd.f32(float %230, float %363, float %478), !dbg !47
  %511 = call float @llvm.fmuladd.f32(float %230, float %365, float %479), !dbg !47
  %512 = call float @llvm.fmuladd.f32(float %230, float %367, float %480), !dbg !47
  %513 = call float @llvm.fmuladd.f32(float %230, float %369, float %481), !dbg !47
  %514 = call float @llvm.fmuladd.f32(float %232, float %363, float %482), !dbg !47
  %515 = call float @llvm.fmuladd.f32(float %232, float %365, float %483), !dbg !47
  %516 = call float @llvm.fmuladd.f32(float %232, float %367, float %484), !dbg !47
  %517 = call float @llvm.fmuladd.f32(float %232, float %369, float %485), !dbg !47
  %518 = call float @llvm.fmuladd.f32(float %234, float %363, float %486), !dbg !47
  %519 = call float @llvm.fmuladd.f32(float %234, float %365, float %487), !dbg !47
  %520 = call float @llvm.fmuladd.f32(float %234, float %367, float %488), !dbg !47
  %521 = call float @llvm.fmuladd.f32(float %234, float %369, float %489), !dbg !47
  %522 = call float @llvm.fmuladd.f32(float %236, float %363, float %490), !dbg !47
  %523 = call float @llvm.fmuladd.f32(float %236, float %365, float %491), !dbg !47
  %524 = call float @llvm.fmuladd.f32(float %236, float %367, float %492), !dbg !47
  %525 = call float @llvm.fmuladd.f32(float %236, float %369, float %493), !dbg !47
  %526 = call float @llvm.fmuladd.f32(float %238, float %363, float %494), !dbg !47
  %527 = call float @llvm.fmuladd.f32(float %238, float %365, float %495), !dbg !47
  %528 = call float @llvm.fmuladd.f32(float %238, float %367, float %496), !dbg !47
  %529 = call float @llvm.fmuladd.f32(float %238, float %369, float %497), !dbg !47
  %530 = call float @llvm.fmuladd.f32(float %240, float %363, float %498), !dbg !47
  %531 = call float @llvm.fmuladd.f32(float %240, float %365, float %499), !dbg !47
  %532 = call float @llvm.fmuladd.f32(float %240, float %367, float %500), !dbg !47
  %533 = call float @llvm.fmuladd.f32(float %240, float %369, float %501), !dbg !47
  %534 = call float @llvm.fmuladd.f32(float %242, float %363, float %502), !dbg !47
  %535 = call float @llvm.fmuladd.f32(float %242, float %365, float %503), !dbg !47
  %536 = call float @llvm.fmuladd.f32(float %242, float %367, float %504), !dbg !47
  %537 = call float @llvm.fmuladd.f32(float %242, float %369, float %505), !dbg !47
  %538 = call float @llvm.fmuladd.f32(float %244, float %371, float %506), !dbg !47
  %539 = call float @llvm.fmuladd.f32(float %244, float %373, float %507), !dbg !47
  %540 = call float @llvm.fmuladd.f32(float %244, float %375, float %508), !dbg !47
  %541 = call float @llvm.fmuladd.f32(float %244, float %377, float %509), !dbg !47
  %542 = call float @llvm.fmuladd.f32(float %246, float %371, float %510), !dbg !47
  %543 = call float @llvm.fmuladd.f32(float %246, float %373, float %511), !dbg !47
  %544 = call float @llvm.fmuladd.f32(float %246, float %375, float %512), !dbg !47
  %545 = call float @llvm.fmuladd.f32(float %246, float %377, float %513), !dbg !47
  %546 = call float @llvm.fmuladd.f32(float %248, float %371, float %514), !dbg !47
  %547 = call float @llvm.fmuladd.f32(float %248, float %373, float %515), !dbg !47
  %548 = call float @llvm.fmuladd.f32(float %248, float %375, float %516), !dbg !47
  %549 = call float @llvm.fmuladd.f32(float %248, float %377, float %517), !dbg !47
  %550 = call float @llvm.fmuladd.f32(float %250, float %371, float %518), !dbg !47
  %551 = call float @llvm.fmuladd.f32(float %250, float %373, float %519), !dbg !47
  %552 = call float @llvm.fmuladd.f32(float %250, float %375, float %520), !dbg !47
  %553 = call float @llvm.fmuladd.f32(float %250, float %377, float %521), !dbg !47
  %554 = call float @llvm.fmuladd.f32(float %252, float %371, float %522), !dbg !47
  %555 = call float @llvm.fmuladd.f32(float %252, float %373, float %523), !dbg !47
  %556 = call float @llvm.fmuladd.f32(float %252, float %375, float %524), !dbg !47
  %557 = call float @llvm.fmuladd.f32(float %252, float %377, float %525), !dbg !47
  %558 = call float @llvm.fmuladd.f32(float %254, float %371, float %526), !dbg !47
  %559 = call float @llvm.fmuladd.f32(float %254, float %373, float %527), !dbg !47
  %560 = call float @llvm.fmuladd.f32(float %254, float %375, float %528), !dbg !47
  %561 = call float @llvm.fmuladd.f32(float %254, float %377, float %529), !dbg !47
  %562 = call float @llvm.fmuladd.f32(float %256, float %371, float %530), !dbg !47
  %563 = call float @llvm.fmuladd.f32(float %256, float %373, float %531), !dbg !47
  %564 = call float @llvm.fmuladd.f32(float %256, float %375, float %532), !dbg !47
  %565 = call float @llvm.fmuladd.f32(float %256, float %377, float %533), !dbg !47
  %566 = call float @llvm.fmuladd.f32(float %258, float %371, float %534), !dbg !47
  %567 = call float @llvm.fmuladd.f32(float %258, float %373, float %535), !dbg !47
  %568 = call float @llvm.fmuladd.f32(float %258, float %375, float %536), !dbg !47
  %569 = call float @llvm.fmuladd.f32(float %258, float %377, float %537), !dbg !47
  %570 = call float @llvm.fmuladd.f32(float %260, float %379, float %538), !dbg !47
  %571 = call float @llvm.fmuladd.f32(float %260, float %381, float %539), !dbg !47
  %572 = call float @llvm.fmuladd.f32(float %260, float %383, float %540), !dbg !47
  %573 = call float @llvm.fmuladd.f32(float %260, float %385, float %541), !dbg !47
  %574 = call float @llvm.fmuladd.f32(float %262, float %379, float %542), !dbg !47
  %575 = call float @llvm.fmuladd.f32(float %262, float %381, float %543), !dbg !47
  %576 = call float @llvm.fmuladd.f32(float %262, float %383, float %544), !dbg !47
  %577 = call float @llvm.fmuladd.f32(float %262, float %385, float %545), !dbg !47
  %578 = call float @llvm.fmuladd.f32(float %264, float %379, float %546), !dbg !47
  %579 = call float @llvm.fmuladd.f32(float %264, float %381, float %547), !dbg !47
  %580 = call float @llvm.fmuladd.f32(float %264, float %383, float %548), !dbg !47
  %581 = call float @llvm.fmuladd.f32(float %264, float %385, float %549), !dbg !47
  %582 = call float @llvm.fmuladd.f32(float %266, float %379, float %550), !dbg !47
  %583 = call float @llvm.fmuladd.f32(float %266, float %381, float %551), !dbg !47
  %584 = call float @llvm.fmuladd.f32(float %266, float %383, float %552), !dbg !47
  %585 = call float @llvm.fmuladd.f32(float %266, float %385, float %553), !dbg !47
  %586 = call float @llvm.fmuladd.f32(float %268, float %379, float %554), !dbg !47
  %587 = call float @llvm.fmuladd.f32(float %268, float %381, float %555), !dbg !47
  %588 = call float @llvm.fmuladd.f32(float %268, float %383, float %556), !dbg !47
  %589 = call float @llvm.fmuladd.f32(float %268, float %385, float %557), !dbg !47
  %590 = call float @llvm.fmuladd.f32(float %270, float %379, float %558), !dbg !47
  %591 = call float @llvm.fmuladd.f32(float %270, float %381, float %559), !dbg !47
  %592 = call float @llvm.fmuladd.f32(float %270, float %383, float %560), !dbg !47
  %593 = call float @llvm.fmuladd.f32(float %270, float %385, float %561), !dbg !47
  %594 = call float @llvm.fmuladd.f32(float %272, float %379, float %562), !dbg !47
  %595 = call float @llvm.fmuladd.f32(float %272, float %381, float %563), !dbg !47
  %596 = call float @llvm.fmuladd.f32(float %272, float %383, float %564), !dbg !47
  %597 = call float @llvm.fmuladd.f32(float %272, float %385, float %565), !dbg !47
  %598 = call float @llvm.fmuladd.f32(float %274, float %379, float %566), !dbg !47
  %599 = call float @llvm.fmuladd.f32(float %274, float %381, float %567), !dbg !47
  %600 = call float @llvm.fmuladd.f32(float %274, float %383, float %568), !dbg !47
  %601 = call float @llvm.fmuladd.f32(float %274, float %385, float %569), !dbg !47
  %602 = call float @llvm.fmuladd.f32(float %276, float %387, float %570), !dbg !47
  %603 = call float @llvm.fmuladd.f32(float %276, float %389, float %571), !dbg !47
  %604 = call float @llvm.fmuladd.f32(float %276, float %391, float %572), !dbg !47
  %605 = call float @llvm.fmuladd.f32(float %276, float %393, float %573), !dbg !47
  %606 = call float @llvm.fmuladd.f32(float %278, float %387, float %574), !dbg !47
  %607 = call float @llvm.fmuladd.f32(float %278, float %389, float %575), !dbg !47
  %608 = call float @llvm.fmuladd.f32(float %278, float %391, float %576), !dbg !47
  %609 = call float @llvm.fmuladd.f32(float %278, float %393, float %577), !dbg !47
  %610 = call float @llvm.fmuladd.f32(float %280, float %387, float %578), !dbg !47
  %611 = call float @llvm.fmuladd.f32(float %280, float %389, float %579), !dbg !47
  %612 = call float @llvm.fmuladd.f32(float %280, float %391, float %580), !dbg !47
  %613 = call float @llvm.fmuladd.f32(float %280, float %393, float %581), !dbg !47
  %614 = call float @llvm.fmuladd.f32(float %282, float %387, float %582), !dbg !47
  %615 = call float @llvm.fmuladd.f32(float %282, float %389, float %583), !dbg !47
  %616 = call float @llvm.fmuladd.f32(float %282, float %391, float %584), !dbg !47
  %617 = call float @llvm.fmuladd.f32(float %282, float %393, float %585), !dbg !47
  %618 = call float @llvm.fmuladd.f32(float %284, float %387, float %586), !dbg !47
  %619 = call float @llvm.fmuladd.f32(float %284, float %389, float %587), !dbg !47
  %620 = call float @llvm.fmuladd.f32(float %284, float %391, float %588), !dbg !47
  %621 = call float @llvm.fmuladd.f32(float %284, float %393, float %589), !dbg !47
  %622 = call float @llvm.fmuladd.f32(float %286, float %387, float %590), !dbg !47
  %623 = call float @llvm.fmuladd.f32(float %286, float %389, float %591), !dbg !47
  %624 = call float @llvm.fmuladd.f32(float %286, float %391, float %592), !dbg !47
  %625 = call float @llvm.fmuladd.f32(float %286, float %393, float %593), !dbg !47
  %626 = call float @llvm.fmuladd.f32(float %288, float %387, float %594), !dbg !47
  %627 = call float @llvm.fmuladd.f32(float %288, float %389, float %595), !dbg !47
  %628 = call float @llvm.fmuladd.f32(float %288, float %391, float %596), !dbg !47
  %629 = call float @llvm.fmuladd.f32(float %288, float %393, float %597), !dbg !47
  %630 = call float @llvm.fmuladd.f32(float %290, float %387, float %598), !dbg !47
  %631 = call float @llvm.fmuladd.f32(float %290, float %389, float %599), !dbg !47
  %632 = call float @llvm.fmuladd.f32(float %290, float %391, float %600), !dbg !47
  %633 = call float @llvm.fmuladd.f32(float %290, float %393, float %601), !dbg !47
  %634 = call float @llvm.fmuladd.f32(float %292, float %395, float %602), !dbg !47
  %635 = call float @llvm.fmuladd.f32(float %292, float %397, float %603), !dbg !47
  %636 = call float @llvm.fmuladd.f32(float %292, float %399, float %604), !dbg !47
  %637 = call float @llvm.fmuladd.f32(float %292, float %401, float %605), !dbg !47
  %638 = call float @llvm.fmuladd.f32(float %294, float %395, float %606), !dbg !47
  %639 = call float @llvm.fmuladd.f32(float %294, float %397, float %607), !dbg !47
  %640 = call float @llvm.fmuladd.f32(float %294, float %399, float %608), !dbg !47
  %641 = call float @llvm.fmuladd.f32(float %294, float %401, float %609), !dbg !47
  %642 = call float @llvm.fmuladd.f32(float %296, float %395, float %610), !dbg !47
  %643 = call float @llvm.fmuladd.f32(float %296, float %397, float %611), !dbg !47
  %644 = call float @llvm.fmuladd.f32(float %296, float %399, float %612), !dbg !47
  %645 = call float @llvm.fmuladd.f32(float %296, float %401, float %613), !dbg !47
  %646 = call float @llvm.fmuladd.f32(float %298, float %395, float %614), !dbg !47
  %647 = call float @llvm.fmuladd.f32(float %298, float %397, float %615), !dbg !47
  %648 = call float @llvm.fmuladd.f32(float %298, float %399, float %616), !dbg !47
  %649 = call float @llvm.fmuladd.f32(float %298, float %401, float %617), !dbg !47
  %650 = call float @llvm.fmuladd.f32(float %300, float %395, float %618), !dbg !47
  %651 = call float @llvm.fmuladd.f32(float %300, float %397, float %619), !dbg !47
  %652 = call float @llvm.fmuladd.f32(float %300, float %399, float %620), !dbg !47
  %653 = call float @llvm.fmuladd.f32(float %300, float %401, float %621), !dbg !47
  %654 = call float @llvm.fmuladd.f32(float %302, float %395, float %622), !dbg !47
  %655 = call float @llvm.fmuladd.f32(float %302, float %397, float %623), !dbg !47
  %656 = call float @llvm.fmuladd.f32(float %302, float %399, float %624), !dbg !47
  %657 = call float @llvm.fmuladd.f32(float %302, float %401, float %625), !dbg !47
  %658 = call float @llvm.fmuladd.f32(float %304, float %395, float %626), !dbg !47
  %659 = call float @llvm.fmuladd.f32(float %304, float %397, float %627), !dbg !47
  %660 = call float @llvm.fmuladd.f32(float %304, float %399, float %628), !dbg !47
  %661 = call float @llvm.fmuladd.f32(float %304, float %401, float %629), !dbg !47
  %662 = call float @llvm.fmuladd.f32(float %306, float %395, float %630), !dbg !47
  %663 = call float @llvm.fmuladd.f32(float %306, float %397, float %631), !dbg !47
  %664 = call float @llvm.fmuladd.f32(float %306, float %399, float %632), !dbg !47
  %665 = call float @llvm.fmuladd.f32(float %306, float %401, float %633), !dbg !47
  %666 = call float @llvm.fmuladd.f32(float %308, float %403, float %634), !dbg !47
  %667 = call float @llvm.fmuladd.f32(float %308, float %405, float %635), !dbg !47
  %668 = call float @llvm.fmuladd.f32(float %308, float %407, float %636), !dbg !47
  %669 = call float @llvm.fmuladd.f32(float %308, float %409, float %637), !dbg !47
  %670 = call float @llvm.fmuladd.f32(float %310, float %403, float %638), !dbg !47
  %671 = call float @llvm.fmuladd.f32(float %310, float %405, float %639), !dbg !47
  %672 = call float @llvm.fmuladd.f32(float %310, float %407, float %640), !dbg !47
  %673 = call float @llvm.fmuladd.f32(float %310, float %409, float %641), !dbg !47
  %674 = call float @llvm.fmuladd.f32(float %312, float %403, float %642), !dbg !47
  %675 = call float @llvm.fmuladd.f32(float %312, float %405, float %643), !dbg !47
  %676 = call float @llvm.fmuladd.f32(float %312, float %407, float %644), !dbg !47
  %677 = call float @llvm.fmuladd.f32(float %312, float %409, float %645), !dbg !47
  %678 = call float @llvm.fmuladd.f32(float %314, float %403, float %646), !dbg !47
  %679 = call float @llvm.fmuladd.f32(float %314, float %405, float %647), !dbg !47
  %680 = call float @llvm.fmuladd.f32(float %314, float %407, float %648), !dbg !47
  %681 = call float @llvm.fmuladd.f32(float %314, float %409, float %649), !dbg !47
  %682 = call float @llvm.fmuladd.f32(float %316, float %403, float %650), !dbg !47
  %683 = call float @llvm.fmuladd.f32(float %316, float %405, float %651), !dbg !47
  %684 = call float @llvm.fmuladd.f32(float %316, float %407, float %652), !dbg !47
  %685 = call float @llvm.fmuladd.f32(float %316, float %409, float %653), !dbg !47
  %686 = call float @llvm.fmuladd.f32(float %318, float %403, float %654), !dbg !47
  %687 = call float @llvm.fmuladd.f32(float %318, float %405, float %655), !dbg !47
  %688 = call float @llvm.fmuladd.f32(float %318, float %407, float %656), !dbg !47
  %689 = call float @llvm.fmuladd.f32(float %318, float %409, float %657), !dbg !47
  %690 = call float @llvm.fmuladd.f32(float %320, float %403, float %658), !dbg !47
  %691 = call float @llvm.fmuladd.f32(float %320, float %405, float %659), !dbg !47
  %692 = call float @llvm.fmuladd.f32(float %320, float %407, float %660), !dbg !47
  %693 = call float @llvm.fmuladd.f32(float %320, float %409, float %661), !dbg !47
  %694 = call float @llvm.fmuladd.f32(float %322, float %403, float %662), !dbg !47
  %695 = call float @llvm.fmuladd.f32(float %322, float %405, float %663), !dbg !47
  %696 = call float @llvm.fmuladd.f32(float %322, float %407, float %664), !dbg !47
  %697 = call float @llvm.fmuladd.f32(float %322, float %409, float %665), !dbg !47
  %698 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } undef, float %666, 0, !dbg !47
  %699 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %698, float %667, 1, !dbg !47
  %700 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %699, float %668, 2, !dbg !47
  %701 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %700, float %669, 3, !dbg !47
  %702 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %701, float %670, 4, !dbg !47
  %703 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %702, float %671, 5, !dbg !47
  %704 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %703, float %672, 6, !dbg !47
  %705 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %704, float %673, 7, !dbg !47
  %706 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %705, float %674, 8, !dbg !47
  %707 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %706, float %675, 9, !dbg !47
  %708 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %707, float %676, 10, !dbg !47
  %709 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %708, float %677, 11, !dbg !47
  %710 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %709, float %678, 12, !dbg !47
  %711 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %710, float %679, 13, !dbg !47
  %712 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %711, float %680, 14, !dbg !47
  %713 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %712, float %681, 15, !dbg !47
  %714 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %713, float %682, 16, !dbg !47
  %715 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %714, float %683, 17, !dbg !47
  %716 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %715, float %684, 18, !dbg !47
  %717 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %716, float %685, 19, !dbg !47
  %718 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %717, float %686, 20, !dbg !47
  %719 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %718, float %687, 21, !dbg !47
  %720 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %719, float %688, 22, !dbg !47
  %721 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %720, float %689, 23, !dbg !47
  %722 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %721, float %690, 24, !dbg !47
  %723 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %722, float %691, 25, !dbg !47
  %724 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %723, float %692, 26, !dbg !47
  %725 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %724, float %693, 27, !dbg !47
  %726 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %725, float %694, 28, !dbg !47
  %727 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %726, float %695, 29, !dbg !47
  %728 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %727, float %696, 30, !dbg !47
  %729 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %728, float %697, 31, !dbg !47
  %730 = extractvalue { ptr addrspace(1), ptr addrspace(1) } %157, 1, !dbg !48
  %731 = getelementptr float, ptr addrspace(1) %161, i32 8, !dbg !48
  %732 = getelementptr float, ptr addrspace(1) %730, i32 8, !dbg !48
  %733 = insertvalue { ptr addrspace(1), ptr addrspace(1) } undef, ptr addrspace(1) %731, 0, !dbg !48
  %734 = insertvalue { ptr addrspace(1), ptr addrspace(1) } %733, ptr addrspace(1) %732, 1, !dbg !48
  %735 = extractvalue { ptr addrspace(1), ptr addrspace(1) } %158, 1, !dbg !49
  %736 = getelementptr float, ptr addrspace(1) %166, i32 %153, !dbg !49
  %737 = getelementptr float, ptr addrspace(1) %735, i32 %153, !dbg !49
  %738 = insertvalue { ptr addrspace(1), ptr addrspace(1) } undef, ptr addrspace(1) %736, 0, !dbg !49
  %739 = insertvalue { ptr addrspace(1), ptr addrspace(1) } %738, ptr addrspace(1) %737, 1, !dbg !49
  %740 = add i32 %155, 1, !dbg !46
  br label %154, !dbg !46

741:                                              ; preds = %154
  %742 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 0, !dbg !50
  %743 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 1, !dbg !50
  %744 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 2, !dbg !50
  %745 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 3, !dbg !50
  %746 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 4, !dbg !50
  %747 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 5, !dbg !50
  %748 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 6, !dbg !50
  %749 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 7, !dbg !50
  %750 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 8, !dbg !50
  %751 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 9, !dbg !50
  %752 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 10, !dbg !50
  %753 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 11, !dbg !50
  %754 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 12, !dbg !50
  %755 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 13, !dbg !50
  %756 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 14, !dbg !50
  %757 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 15, !dbg !50
  %758 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 16, !dbg !50
  %759 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 17, !dbg !50
  %760 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 18, !dbg !50
  %761 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 19, !dbg !50
  %762 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 20, !dbg !50
  %763 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 21, !dbg !50
  %764 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 22, !dbg !50
  %765 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 23, !dbg !50
  %766 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 24, !dbg !50
  %767 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 25, !dbg !50
  %768 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 26, !dbg !50
  %769 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 27, !dbg !50
  %770 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 28, !dbg !50
  %771 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 29, !dbg !50
  %772 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 30, !dbg !50
  %773 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %156, 31, !dbg !50
  %774 = fptrunc float %742 to half, !dbg !50
  %775 = fptrunc float %743 to half, !dbg !50
  %776 = fptrunc float %744 to half, !dbg !50
  %777 = fptrunc float %745 to half, !dbg !50
  %778 = fptrunc float %746 to half, !dbg !50
  %779 = fptrunc float %747 to half, !dbg !50
  %780 = fptrunc float %748 to half, !dbg !50
  %781 = fptrunc float %749 to half, !dbg !50
  %782 = fptrunc float %750 to half, !dbg !50
  %783 = fptrunc float %751 to half, !dbg !50
  %784 = fptrunc float %752 to half, !dbg !50
  %785 = fptrunc float %753 to half, !dbg !50
  %786 = fptrunc float %754 to half, !dbg !50
  %787 = fptrunc float %755 to half, !dbg !50
  %788 = fptrunc float %756 to half, !dbg !50
  %789 = fptrunc float %757 to half, !dbg !50
  %790 = fptrunc float %758 to half, !dbg !50
  %791 = fptrunc float %759 to half, !dbg !50
  %792 = fptrunc float %760 to half, !dbg !50
  %793 = fptrunc float %761 to half, !dbg !50
  %794 = fptrunc float %762 to half, !dbg !50
  %795 = fptrunc float %763 to half, !dbg !50
  %796 = fptrunc float %764 to half, !dbg !50
  %797 = fptrunc float %765 to half, !dbg !50
  %798 = fptrunc float %766 to half, !dbg !50
  %799 = fptrunc float %767 to half, !dbg !50
  %800 = fptrunc float %768 to half, !dbg !50
  %801 = fptrunc float %769 to half, !dbg !50
  %802 = fptrunc float %770 to half, !dbg !50
  %803 = fptrunc float %771 to half, !dbg !50
  %804 = fptrunc float %772 to half, !dbg !50
  %805 = fptrunc float %773 to half, !dbg !50
  %806 = mul i32 %8, %116, !dbg !51
  %807 = mul i32 %8, %117, !dbg !51
  %808 = mul i32 %8, %118, !dbg !51
  %809 = mul i32 %8, %119, !dbg !51
  %810 = mul i32 %8, %120, !dbg !51
  %811 = mul i32 %8, %121, !dbg !51
  %812 = mul i32 %8, %122, !dbg !51
  %813 = mul i32 %8, %123, !dbg !51
  %814 = getelementptr float, ptr addrspace(1) %2, i32 %806, !dbg !52
  %815 = getelementptr float, ptr addrspace(1) %2, i32 %807, !dbg !52
  %816 = getelementptr float, ptr addrspace(1) %2, i32 %808, !dbg !52
  %817 = getelementptr float, ptr addrspace(1) %2, i32 %809, !dbg !52
  %818 = getelementptr float, ptr addrspace(1) %2, i32 %810, !dbg !52
  %819 = getelementptr float, ptr addrspace(1) %2, i32 %811, !dbg !52
  %820 = getelementptr float, ptr addrspace(1) %2, i32 %812, !dbg !52
  %821 = getelementptr float, ptr addrspace(1) %2, i32 %813, !dbg !52
  %822 = getelementptr float, ptr addrspace(1) %814, i32 %128, !dbg !53
  %823 = getelementptr float, ptr addrspace(1) %814, i32 %129, !dbg !53
  %824 = getelementptr float, ptr addrspace(1) %814, i32 %130, !dbg !53
  %825 = getelementptr float, ptr addrspace(1) %814, i32 %131, !dbg !53
  %826 = getelementptr float, ptr addrspace(1) %815, i32 %128, !dbg !53
  %827 = getelementptr float, ptr addrspace(1) %815, i32 %129, !dbg !53
  %828 = getelementptr float, ptr addrspace(1) %815, i32 %130, !dbg !53
  %829 = getelementptr float, ptr addrspace(1) %815, i32 %131, !dbg !53
  %830 = getelementptr float, ptr addrspace(1) %816, i32 %128, !dbg !53
  %831 = getelementptr float, ptr addrspace(1) %816, i32 %129, !dbg !53
  %832 = getelementptr float, ptr addrspace(1) %816, i32 %130, !dbg !53
  %833 = getelementptr float, ptr addrspace(1) %816, i32 %131, !dbg !53
  %834 = getelementptr float, ptr addrspace(1) %817, i32 %128, !dbg !53
  %835 = getelementptr float, ptr addrspace(1) %817, i32 %129, !dbg !53
  %836 = getelementptr float, ptr addrspace(1) %817, i32 %130, !dbg !53
  %837 = getelementptr float, ptr addrspace(1) %817, i32 %131, !dbg !53
  %838 = getelementptr float, ptr addrspace(1) %818, i32 %128, !dbg !53
  %839 = getelementptr float, ptr addrspace(1) %818, i32 %129, !dbg !53
  %840 = getelementptr float, ptr addrspace(1) %818, i32 %130, !dbg !53
  %841 = getelementptr float, ptr addrspace(1) %818, i32 %131, !dbg !53
  %842 = getelementptr float, ptr addrspace(1) %819, i32 %128, !dbg !53
  %843 = getelementptr float, ptr addrspace(1) %819, i32 %129, !dbg !53
  %844 = getelementptr float, ptr addrspace(1) %819, i32 %130, !dbg !53
  %845 = getelementptr float, ptr addrspace(1) %819, i32 %131, !dbg !53
  %846 = getelementptr float, ptr addrspace(1) %820, i32 %128, !dbg !53
  %847 = getelementptr float, ptr addrspace(1) %820, i32 %129, !dbg !53
  %848 = getelementptr float, ptr addrspace(1) %820, i32 %130, !dbg !53
  %849 = getelementptr float, ptr addrspace(1) %820, i32 %131, !dbg !53
  %850 = getelementptr float, ptr addrspace(1) %821, i32 %128, !dbg !53
  %851 = getelementptr float, ptr addrspace(1) %821, i32 %129, !dbg !53
  %852 = getelementptr float, ptr addrspace(1) %821, i32 %130, !dbg !53
  %853 = getelementptr float, ptr addrspace(1) %821, i32 %131, !dbg !53
  %854 = icmp slt i32 %116, %3, !dbg !54
  %855 = icmp slt i32 %117, %3, !dbg !54
  %856 = icmp slt i32 %118, %3, !dbg !54
  %857 = icmp slt i32 %119, %3, !dbg !54
  %858 = icmp slt i32 %120, %3, !dbg !54
  %859 = icmp slt i32 %121, %3, !dbg !54
  %860 = icmp slt i32 %122, %3, !dbg !54
  %861 = icmp slt i32 %123, %3, !dbg !54
  %862 = icmp slt i32 %128, %4, !dbg !55
  %863 = and i1 %854, %862, !dbg !56
  %864 = and i1 %855, %862, !dbg !56
  %865 = and i1 %856, %862, !dbg !56
  %866 = and i1 %857, %862, !dbg !56
  %867 = and i1 %858, %862, !dbg !56
  %868 = and i1 %859, %862, !dbg !56
  %869 = and i1 %860, %862, !dbg !56
  %870 = and i1 %861, %862, !dbg !56
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %871 = add i32 %84, 0, !dbg !12
  %872 = add i32 %54, 0, !dbg !12
  %873 = mul i32 %871, 132, !dbg !12
  %874 = add i32 %873, %872, !dbg !12
  %875 = getelementptr half, ptr addrspace(3) @global_smem, i32 %874, !dbg !12
  %876 = insertelement <4 x half> undef, half %774, i32 0, !dbg !12
  %877 = insertelement <4 x half> %876, half %775, i32 1, !dbg !12
  %878 = insertelement <4 x half> %877, half %776, i32 2, !dbg !12
  %879 = insertelement <4 x half> %878, half %777, i32 3, !dbg !12
  store <4 x half> %879, ptr addrspace(3) %875, align 8, !dbg !12
  %880 = add i32 %84, 1, !dbg !12
  %881 = mul i32 %880, 132, !dbg !12
  %882 = add i32 %881, %872, !dbg !12
  %883 = getelementptr half, ptr addrspace(3) @global_smem, i32 %882, !dbg !12
  %884 = insertelement <4 x half> undef, half %778, i32 0, !dbg !12
  %885 = insertelement <4 x half> %884, half %779, i32 1, !dbg !12
  %886 = insertelement <4 x half> %885, half %780, i32 2, !dbg !12
  %887 = insertelement <4 x half> %886, half %781, i32 3, !dbg !12
  store <4 x half> %887, ptr addrspace(3) %883, align 8, !dbg !12
  %888 = add i32 %84, 2, !dbg !12
  %889 = mul i32 %888, 132, !dbg !12
  %890 = add i32 %889, %872, !dbg !12
  %891 = getelementptr half, ptr addrspace(3) @global_smem, i32 %890, !dbg !12
  %892 = insertelement <4 x half> undef, half %782, i32 0, !dbg !12
  %893 = insertelement <4 x half> %892, half %783, i32 1, !dbg !12
  %894 = insertelement <4 x half> %893, half %784, i32 2, !dbg !12
  %895 = insertelement <4 x half> %894, half %785, i32 3, !dbg !12
  store <4 x half> %895, ptr addrspace(3) %891, align 8, !dbg !12
  %896 = add i32 %84, 3, !dbg !12
  %897 = mul i32 %896, 132, !dbg !12
  %898 = add i32 %897, %872, !dbg !12
  %899 = getelementptr half, ptr addrspace(3) @global_smem, i32 %898, !dbg !12
  %900 = insertelement <4 x half> undef, half %786, i32 0, !dbg !12
  %901 = insertelement <4 x half> %900, half %787, i32 1, !dbg !12
  %902 = insertelement <4 x half> %901, half %788, i32 2, !dbg !12
  %903 = insertelement <4 x half> %902, half %789, i32 3, !dbg !12
  store <4 x half> %903, ptr addrspace(3) %899, align 8, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %904 = add i32 %40, 0, !dbg !12
  %905 = mul i32 %904, 132, !dbg !12
  %906 = add i32 %905, %872, !dbg !12
  %907 = getelementptr half, ptr addrspace(3) @global_smem, i32 %906, !dbg !12
  %908 = load <4 x half>, ptr addrspace(3) %907, align 8, !dbg !12
  %909 = extractelement <4 x half> %908, i32 0, !dbg !12
  %910 = extractelement <4 x half> %908, i32 1, !dbg !12
  %911 = extractelement <4 x half> %908, i32 2, !dbg !12
  %912 = extractelement <4 x half> %908, i32 3, !dbg !12
  %913 = add i32 %40, 16, !dbg !12
  %914 = mul i32 %913, 132, !dbg !12
  %915 = add i32 %914, %872, !dbg !12
  %916 = getelementptr half, ptr addrspace(3) @global_smem, i32 %915, !dbg !12
  %917 = load <4 x half>, ptr addrspace(3) %916, align 8, !dbg !12
  %918 = extractelement <4 x half> %917, i32 0, !dbg !12
  %919 = extractelement <4 x half> %917, i32 1, !dbg !12
  %920 = extractelement <4 x half> %917, i32 2, !dbg !12
  %921 = extractelement <4 x half> %917, i32 3, !dbg !12
  %922 = add i32 %40, 32, !dbg !12
  %923 = mul i32 %922, 132, !dbg !12
  %924 = add i32 %923, %872, !dbg !12
  %925 = getelementptr half, ptr addrspace(3) @global_smem, i32 %924, !dbg !12
  %926 = load <4 x half>, ptr addrspace(3) %925, align 8, !dbg !12
  %927 = extractelement <4 x half> %926, i32 0, !dbg !12
  %928 = extractelement <4 x half> %926, i32 1, !dbg !12
  %929 = extractelement <4 x half> %926, i32 2, !dbg !12
  %930 = extractelement <4 x half> %926, i32 3, !dbg !12
  %931 = add i32 %40, 48, !dbg !12
  %932 = mul i32 %931, 132, !dbg !12
  %933 = add i32 %932, %872, !dbg !12
  %934 = getelementptr half, ptr addrspace(3) @global_smem, i32 %933, !dbg !12
  %935 = load <4 x half>, ptr addrspace(3) %934, align 8, !dbg !12
  %936 = extractelement <4 x half> %935, i32 0, !dbg !12
  %937 = extractelement <4 x half> %935, i32 1, !dbg !12
  %938 = extractelement <4 x half> %935, i32 2, !dbg !12
  %939 = extractelement <4 x half> %935, i32 3, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %940 = insertelement <4 x half> undef, half %790, i32 0, !dbg !12
  %941 = insertelement <4 x half> %940, half %791, i32 1, !dbg !12
  %942 = insertelement <4 x half> %941, half %792, i32 2, !dbg !12
  %943 = insertelement <4 x half> %942, half %793, i32 3, !dbg !12
  store <4 x half> %943, ptr addrspace(3) %875, align 8, !dbg !12
  %944 = insertelement <4 x half> undef, half %794, i32 0, !dbg !12
  %945 = insertelement <4 x half> %944, half %795, i32 1, !dbg !12
  %946 = insertelement <4 x half> %945, half %796, i32 2, !dbg !12
  %947 = insertelement <4 x half> %946, half %797, i32 3, !dbg !12
  store <4 x half> %947, ptr addrspace(3) %883, align 8, !dbg !12
  %948 = insertelement <4 x half> undef, half %798, i32 0, !dbg !12
  %949 = insertelement <4 x half> %948, half %799, i32 1, !dbg !12
  %950 = insertelement <4 x half> %949, half %800, i32 2, !dbg !12
  %951 = insertelement <4 x half> %950, half %801, i32 3, !dbg !12
  store <4 x half> %951, ptr addrspace(3) %891, align 8, !dbg !12
  %952 = insertelement <4 x half> undef, half %802, i32 0, !dbg !12
  %953 = insertelement <4 x half> %952, half %803, i32 1, !dbg !12
  %954 = insertelement <4 x half> %953, half %804, i32 2, !dbg !12
  %955 = insertelement <4 x half> %954, half %805, i32 3, !dbg !12
  store <4 x half> %955, ptr addrspace(3) %899, align 8, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %956 = load <4 x half>, ptr addrspace(3) %907, align 8, !dbg !12
  %957 = extractelement <4 x half> %956, i32 0, !dbg !12
  %958 = extractelement <4 x half> %956, i32 1, !dbg !12
  %959 = extractelement <4 x half> %956, i32 2, !dbg !12
  %960 = extractelement <4 x half> %956, i32 3, !dbg !12
  %961 = load <4 x half>, ptr addrspace(3) %916, align 8, !dbg !12
  %962 = extractelement <4 x half> %961, i32 0, !dbg !12
  %963 = extractelement <4 x half> %961, i32 1, !dbg !12
  %964 = extractelement <4 x half> %961, i32 2, !dbg !12
  %965 = extractelement <4 x half> %961, i32 3, !dbg !12
  %966 = load <4 x half>, ptr addrspace(3) %925, align 8, !dbg !12
  %967 = extractelement <4 x half> %966, i32 0, !dbg !12
  %968 = extractelement <4 x half> %966, i32 1, !dbg !12
  %969 = extractelement <4 x half> %966, i32 2, !dbg !12
  %970 = extractelement <4 x half> %966, i32 3, !dbg !12
  %971 = load <4 x half>, ptr addrspace(3) %934, align 8, !dbg !12
  %972 = extractelement <4 x half> %971, i32 0, !dbg !12
  %973 = extractelement <4 x half> %971, i32 1, !dbg !12
  %974 = extractelement <4 x half> %971, i32 2, !dbg !12
  %975 = extractelement <4 x half> %971, i32 3, !dbg !12
  %976 = fpext half %909 to float, !dbg !12
  %977 = fpext half %910 to float, !dbg !12
  %978 = fpext half %911 to float, !dbg !12
  %979 = fpext half %912 to float, !dbg !12
  %980 = fpext half %918 to float, !dbg !12
  %981 = fpext half %919 to float, !dbg !12
  %982 = fpext half %920 to float, !dbg !12
  %983 = fpext half %921 to float, !dbg !12
  %984 = fpext half %927 to float, !dbg !12
  %985 = fpext half %928 to float, !dbg !12
  %986 = fpext half %929 to float, !dbg !12
  %987 = fpext half %930 to float, !dbg !12
  %988 = fpext half %936 to float, !dbg !12
  %989 = fpext half %937 to float, !dbg !12
  %990 = fpext half %938 to float, !dbg !12
  %991 = fpext half %939 to float, !dbg !12
  %992 = fpext half %957 to float, !dbg !12
  %993 = fpext half %958 to float, !dbg !12
  %994 = fpext half %959 to float, !dbg !12
  %995 = fpext half %960 to float, !dbg !12
  %996 = fpext half %962 to float, !dbg !12
  %997 = fpext half %963 to float, !dbg !12
  %998 = fpext half %964 to float, !dbg !12
  %999 = fpext half %965 to float, !dbg !12
  %1000 = fpext half %967 to float, !dbg !12
  %1001 = fpext half %968 to float, !dbg !12
  %1002 = fpext half %969 to float, !dbg !12
  %1003 = fpext half %970 to float, !dbg !12
  %1004 = fpext half %972 to float, !dbg !12
  %1005 = fpext half %973 to float, !dbg !12
  %1006 = fpext half %974 to float, !dbg !12
  %1007 = fpext half %975 to float, !dbg !12
  %1008 = insertelement <1 x float> undef, float %976, i32 0, !dbg !12
  %1009 = bitcast <1 x float> %1008 to i32, !dbg !12
  %1010 = and i1 true, %863, !dbg !12
  br i1 %1010, label %1011, label %1012, !dbg !12

1011:                                             ; preds = %741
  store i32 %1009, ptr addrspace(1) %822, align 4, !dbg !12
  br label %1012, !dbg !12

1012:                                             ; preds = %1011, %741
  %1013 = insertelement <1 x float> undef, float %977, i32 0, !dbg !12
  %1014 = bitcast <1 x float> %1013 to i32, !dbg !12
  br i1 %1010, label %1015, label %1016, !dbg !12

1015:                                             ; preds = %1012
  store i32 %1014, ptr addrspace(1) %823, align 4, !dbg !12
  br label %1016, !dbg !12

1016:                                             ; preds = %1015, %1012
  %1017 = insertelement <1 x float> undef, float %978, i32 0, !dbg !12
  %1018 = bitcast <1 x float> %1017 to i32, !dbg !12
  br i1 %1010, label %1019, label %1020, !dbg !12

1019:                                             ; preds = %1016
  store i32 %1018, ptr addrspace(1) %824, align 4, !dbg !12
  br label %1020, !dbg !12

1020:                                             ; preds = %1019, %1016
  %1021 = insertelement <1 x float> undef, float %979, i32 0, !dbg !12
  %1022 = bitcast <1 x float> %1021 to i32, !dbg !12
  br i1 %1010, label %1023, label %1024, !dbg !12

1023:                                             ; preds = %1020
  store i32 %1022, ptr addrspace(1) %825, align 4, !dbg !12
  br label %1024, !dbg !12

1024:                                             ; preds = %1023, %1020
  %1025 = insertelement <1 x float> undef, float %980, i32 0, !dbg !12
  %1026 = bitcast <1 x float> %1025 to i32, !dbg !12
  %1027 = and i1 true, %864, !dbg !12
  br i1 %1027, label %1028, label %1029, !dbg !12

1028:                                             ; preds = %1024
  store i32 %1026, ptr addrspace(1) %826, align 4, !dbg !12
  br label %1029, !dbg !12

1029:                                             ; preds = %1028, %1024
  %1030 = insertelement <1 x float> undef, float %981, i32 0, !dbg !12
  %1031 = bitcast <1 x float> %1030 to i32, !dbg !12
  br i1 %1027, label %1032, label %1033, !dbg !12

1032:                                             ; preds = %1029
  store i32 %1031, ptr addrspace(1) %827, align 4, !dbg !12
  br label %1033, !dbg !12

1033:                                             ; preds = %1032, %1029
  %1034 = insertelement <1 x float> undef, float %982, i32 0, !dbg !12
  %1035 = bitcast <1 x float> %1034 to i32, !dbg !12
  br i1 %1027, label %1036, label %1037, !dbg !12

1036:                                             ; preds = %1033
  store i32 %1035, ptr addrspace(1) %828, align 4, !dbg !12
  br label %1037, !dbg !12

1037:                                             ; preds = %1036, %1033
  %1038 = insertelement <1 x float> undef, float %983, i32 0, !dbg !12
  %1039 = bitcast <1 x float> %1038 to i32, !dbg !12
  br i1 %1027, label %1040, label %1041, !dbg !12

1040:                                             ; preds = %1037
  store i32 %1039, ptr addrspace(1) %829, align 4, !dbg !12
  br label %1041, !dbg !12

1041:                                             ; preds = %1040, %1037
  %1042 = insertelement <1 x float> undef, float %984, i32 0, !dbg !12
  %1043 = bitcast <1 x float> %1042 to i32, !dbg !12
  %1044 = and i1 true, %865, !dbg !12
  br i1 %1044, label %1045, label %1046, !dbg !12

1045:                                             ; preds = %1041
  store i32 %1043, ptr addrspace(1) %830, align 4, !dbg !12
  br label %1046, !dbg !12

1046:                                             ; preds = %1045, %1041
  %1047 = insertelement <1 x float> undef, float %985, i32 0, !dbg !12
  %1048 = bitcast <1 x float> %1047 to i32, !dbg !12
  br i1 %1044, label %1049, label %1050, !dbg !12

1049:                                             ; preds = %1046
  store i32 %1048, ptr addrspace(1) %831, align 4, !dbg !12
  br label %1050, !dbg !12

1050:                                             ; preds = %1049, %1046
  %1051 = insertelement <1 x float> undef, float %986, i32 0, !dbg !12
  %1052 = bitcast <1 x float> %1051 to i32, !dbg !12
  br i1 %1044, label %1053, label %1054, !dbg !12

1053:                                             ; preds = %1050
  store i32 %1052, ptr addrspace(1) %832, align 4, !dbg !12
  br label %1054, !dbg !12

1054:                                             ; preds = %1053, %1050
  %1055 = insertelement <1 x float> undef, float %987, i32 0, !dbg !12
  %1056 = bitcast <1 x float> %1055 to i32, !dbg !12
  br i1 %1044, label %1057, label %1058, !dbg !12

1057:                                             ; preds = %1054
  store i32 %1056, ptr addrspace(1) %833, align 4, !dbg !12
  br label %1058, !dbg !12

1058:                                             ; preds = %1057, %1054
  %1059 = insertelement <1 x float> undef, float %988, i32 0, !dbg !12
  %1060 = bitcast <1 x float> %1059 to i32, !dbg !12
  %1061 = and i1 true, %866, !dbg !12
  br i1 %1061, label %1062, label %1063, !dbg !12

1062:                                             ; preds = %1058
  store i32 %1060, ptr addrspace(1) %834, align 4, !dbg !12
  br label %1063, !dbg !12

1063:                                             ; preds = %1062, %1058
  %1064 = insertelement <1 x float> undef, float %989, i32 0, !dbg !12
  %1065 = bitcast <1 x float> %1064 to i32, !dbg !12
  br i1 %1061, label %1066, label %1067, !dbg !12

1066:                                             ; preds = %1063
  store i32 %1065, ptr addrspace(1) %835, align 4, !dbg !12
  br label %1067, !dbg !12

1067:                                             ; preds = %1066, %1063
  %1068 = insertelement <1 x float> undef, float %990, i32 0, !dbg !12
  %1069 = bitcast <1 x float> %1068 to i32, !dbg !12
  br i1 %1061, label %1070, label %1071, !dbg !12

1070:                                             ; preds = %1067
  store i32 %1069, ptr addrspace(1) %836, align 4, !dbg !12
  br label %1071, !dbg !12

1071:                                             ; preds = %1070, %1067
  %1072 = insertelement <1 x float> undef, float %991, i32 0, !dbg !12
  %1073 = bitcast <1 x float> %1072 to i32, !dbg !12
  br i1 %1061, label %1074, label %1075, !dbg !12

1074:                                             ; preds = %1071
  store i32 %1073, ptr addrspace(1) %837, align 4, !dbg !12
  br label %1075, !dbg !12

1075:                                             ; preds = %1074, %1071
  %1076 = insertelement <1 x float> undef, float %992, i32 0, !dbg !12
  %1077 = bitcast <1 x float> %1076 to i32, !dbg !12
  %1078 = and i1 true, %867, !dbg !12
  br i1 %1078, label %1079, label %1080, !dbg !12

1079:                                             ; preds = %1075
  store i32 %1077, ptr addrspace(1) %838, align 4, !dbg !12
  br label %1080, !dbg !12

1080:                                             ; preds = %1079, %1075
  %1081 = insertelement <1 x float> undef, float %993, i32 0, !dbg !12
  %1082 = bitcast <1 x float> %1081 to i32, !dbg !12
  br i1 %1078, label %1083, label %1084, !dbg !12

1083:                                             ; preds = %1080
  store i32 %1082, ptr addrspace(1) %839, align 4, !dbg !12
  br label %1084, !dbg !12

1084:                                             ; preds = %1083, %1080
  %1085 = insertelement <1 x float> undef, float %994, i32 0, !dbg !12
  %1086 = bitcast <1 x float> %1085 to i32, !dbg !12
  br i1 %1078, label %1087, label %1088, !dbg !12

1087:                                             ; preds = %1084
  store i32 %1086, ptr addrspace(1) %840, align 4, !dbg !12
  br label %1088, !dbg !12

1088:                                             ; preds = %1087, %1084
  %1089 = insertelement <1 x float> undef, float %995, i32 0, !dbg !12
  %1090 = bitcast <1 x float> %1089 to i32, !dbg !12
  br i1 %1078, label %1091, label %1092, !dbg !12

1091:                                             ; preds = %1088
  store i32 %1090, ptr addrspace(1) %841, align 4, !dbg !12
  br label %1092, !dbg !12

1092:                                             ; preds = %1091, %1088
  %1093 = insertelement <1 x float> undef, float %996, i32 0, !dbg !12
  %1094 = bitcast <1 x float> %1093 to i32, !dbg !12
  %1095 = and i1 true, %868, !dbg !12
  br i1 %1095, label %1096, label %1097, !dbg !12

1096:                                             ; preds = %1092
  store i32 %1094, ptr addrspace(1) %842, align 4, !dbg !12
  br label %1097, !dbg !12

1097:                                             ; preds = %1096, %1092
  %1098 = insertelement <1 x float> undef, float %997, i32 0, !dbg !12
  %1099 = bitcast <1 x float> %1098 to i32, !dbg !12
  br i1 %1095, label %1100, label %1101, !dbg !12

1100:                                             ; preds = %1097
  store i32 %1099, ptr addrspace(1) %843, align 4, !dbg !12
  br label %1101, !dbg !12

1101:                                             ; preds = %1100, %1097
  %1102 = insertelement <1 x float> undef, float %998, i32 0, !dbg !12
  %1103 = bitcast <1 x float> %1102 to i32, !dbg !12
  br i1 %1095, label %1104, label %1105, !dbg !12

1104:                                             ; preds = %1101
  store i32 %1103, ptr addrspace(1) %844, align 4, !dbg !12
  br label %1105, !dbg !12

1105:                                             ; preds = %1104, %1101
  %1106 = insertelement <1 x float> undef, float %999, i32 0, !dbg !12
  %1107 = bitcast <1 x float> %1106 to i32, !dbg !12
  br i1 %1095, label %1108, label %1109, !dbg !12

1108:                                             ; preds = %1105
  store i32 %1107, ptr addrspace(1) %845, align 4, !dbg !12
  br label %1109, !dbg !12

1109:                                             ; preds = %1108, %1105
  %1110 = insertelement <1 x float> undef, float %1000, i32 0, !dbg !12
  %1111 = bitcast <1 x float> %1110 to i32, !dbg !12
  %1112 = and i1 true, %869, !dbg !12
  br i1 %1112, label %1113, label %1114, !dbg !12

1113:                                             ; preds = %1109
  store i32 %1111, ptr addrspace(1) %846, align 4, !dbg !12
  br label %1114, !dbg !12

1114:                                             ; preds = %1113, %1109
  %1115 = insertelement <1 x float> undef, float %1001, i32 0, !dbg !12
  %1116 = bitcast <1 x float> %1115 to i32, !dbg !12
  br i1 %1112, label %1117, label %1118, !dbg !12

1117:                                             ; preds = %1114
  store i32 %1116, ptr addrspace(1) %847, align 4, !dbg !12
  br label %1118, !dbg !12

1118:                                             ; preds = %1117, %1114
  %1119 = insertelement <1 x float> undef, float %1002, i32 0, !dbg !12
  %1120 = bitcast <1 x float> %1119 to i32, !dbg !12
  br i1 %1112, label %1121, label %1122, !dbg !12

1121:                                             ; preds = %1118
  store i32 %1120, ptr addrspace(1) %848, align 4, !dbg !12
  br label %1122, !dbg !12

1122:                                             ; preds = %1121, %1118
  %1123 = insertelement <1 x float> undef, float %1003, i32 0, !dbg !12
  %1124 = bitcast <1 x float> %1123 to i32, !dbg !12
  br i1 %1112, label %1125, label %1126, !dbg !12

1125:                                             ; preds = %1122
  store i32 %1124, ptr addrspace(1) %849, align 4, !dbg !12
  br label %1126, !dbg !12

1126:                                             ; preds = %1125, %1122
  %1127 = insertelement <1 x float> undef, float %1004, i32 0, !dbg !12
  %1128 = bitcast <1 x float> %1127 to i32, !dbg !12
  %1129 = and i1 true, %870, !dbg !12
  br i1 %1129, label %1130, label %1131, !dbg !12

1130:                                             ; preds = %1126
  store i32 %1128, ptr addrspace(1) %850, align 4, !dbg !12
  br label %1131, !dbg !12

1131:                                             ; preds = %1130, %1126
  %1132 = insertelement <1 x float> undef, float %1005, i32 0, !dbg !12
  %1133 = bitcast <1 x float> %1132 to i32, !dbg !12
  br i1 %1129, label %1134, label %1135, !dbg !12

1134:                                             ; preds = %1131
  store i32 %1133, ptr addrspace(1) %851, align 4, !dbg !12
  br label %1135, !dbg !12

1135:                                             ; preds = %1134, %1131
  %1136 = insertelement <1 x float> undef, float %1006, i32 0, !dbg !12
  %1137 = bitcast <1 x float> %1136 to i32, !dbg !12
  br i1 %1129, label %1138, label %1139, !dbg !12

1138:                                             ; preds = %1135
  store i32 %1137, ptr addrspace(1) %852, align 4, !dbg !12
  br label %1139, !dbg !12

1139:                                             ; preds = %1138, %1135
  %1140 = insertelement <1 x float> undef, float %1007, i32 0, !dbg !12
  %1141 = bitcast <1 x float> %1140 to i32, !dbg !12
  br i1 %1129, label %1142, label %1143, !dbg !12

1142:                                             ; preds = %1139
  store i32 %1141, ptr addrspace(1) %853, align 4, !dbg !12
  br label %1143, !dbg !12

1143:                                             ; preds = %1142, %1139
  ret void, !dbg !57
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workgroup.id.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32 %0, i32 %1) #0

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float %0, float %1, float %2) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nofree nounwind willreturn }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}
!nvvm.annotations = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "03-matrix-multiplication.py", directory: "")
!3 = !{ptr @matmul_kernel_0d1d2d3de4de5de6de7c8de9c10de11c, !"kernel", i32 1}
!4 = distinct !DISubprogram(name: "matmul_kernel_0d1d2d3de4de5de6de7c8de9c10de11c", linkageName: "matmul_kernel_0d1d2d3de4de5de6de7c8de9c10de11c", scope: !2, file: !2, line: 168, type: !5, scopeLine: 168, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{}
!7 = !DILocation(line: 214, column: 51, scope: !4)
!8 = !DILocation(line: 216, column: 53, scope: !4)
!9 = !DILocation(line: 217, column: 22, scope: !4)
!10 = !DILocation(line: 229, column: 24, scope: !4)
!11 = !DILocation(line: 230, column: 24, scope: !4)
!12 = !DILocation(line: 251, column: 21, scope: !4)
!13 = !DILocation(line: 192, column: 10, scope: !4)
!14 = !DILocation(line: 21, scope: !15, inlinedAt: !17)
!15 = distinct !DILexicalBlockFile(scope: !4, file: !16, discriminator: 0)
!16 = !DIFile(filename: "standard.py", directory: "/home/pangyunfei/triton_rocm/triton/python/triton/language")
!17 = !DILocation(line: 193, column: 27, scope: !15)
!18 = !DILocation(line: 21, column: 28, scope: !15, inlinedAt: !17)
!19 = !DILocation(line: 21, scope: !15, inlinedAt: !20)
!20 = !DILocation(line: 194, column: 27, scope: !15)
!21 = !DILocation(line: 21, column: 28, scope: !15, inlinedAt: !20)
!22 = !DILocation(line: 199, column: 42, scope: !4)
!23 = !DILocation(line: 200, column: 26, scope: !4)
!24 = !DILocation(line: 201, column: 33, scope: !4)
!25 = !DILocation(line: 202, column: 39, scope: !4)
!26 = !DILocation(line: 111, column: 23, scope: !15, inlinedAt: !27)
!27 = !DILocation(line: 202, column: 52, scope: !15)
!28 = !DILocation(line: 203, column: 37, scope: !4)
!29 = !DILocation(line: 203, column: 31, scope: !4)
!30 = !DILocation(line: 204, column: 23, scope: !4)
!31 = !DILocation(line: 204, column: 44, scope: !4)
!32 = !DILocation(line: 214, column: 23, scope: !4)
!33 = !DILocation(line: 214, column: 38, scope: !4)
!34 = !DILocation(line: 214, column: 68, scope: !4)
!35 = !DILocation(line: 215, column: 23, scope: !4)
!36 = !DILocation(line: 215, column: 38, scope: !4)
!37 = !DILocation(line: 215, column: 68, scope: !4)
!38 = !DILocation(line: 216, column: 41, scope: !4)
!39 = !DILocation(line: 216, column: 22, scope: !4)
!40 = !DILocation(line: 217, column: 40, scope: !4)
!41 = !DILocation(line: 217, column: 52, scope: !4)
!42 = !DILocation(line: 21, scope: !15, inlinedAt: !43)
!43 = !DILocation(line: 225, column: 33, scope: !15)
!44 = !DILocation(line: 21, column: 28, scope: !15, inlinedAt: !43)
!45 = !DILocation(line: 238, column: 33, scope: !4)
!46 = !DILocation(line: 225, column: 22, scope: !4)
!47 = !DILocation(line: 235, scope: !4)
!48 = !DILocation(line: 237, column: 18, scope: !4)
!49 = !DILocation(line: 238, column: 18, scope: !4)
!50 = !DILocation(line: 243, column: 23, scope: !4)
!51 = !DILocation(line: 249, column: 33, scope: !4)
!52 = !DILocation(line: 249, column: 21, scope: !4)
!53 = !DILocation(line: 249, column: 52, scope: !4)
!54 = !DILocation(line: 250, column: 33, scope: !4)
!55 = !DILocation(line: 250, column: 58, scope: !4)
!56 = !DILocation(line: 250, column: 39, scope: !4)
!57 = !DILocation(line: 251, column: 4, scope: !4)