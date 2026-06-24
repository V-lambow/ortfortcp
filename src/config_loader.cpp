#include "config_loader.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <json/json.h>

// ===================== 内部工具函数（命名空间隔离） =====================
namespace JsonParser {
    // 字段存在性+类型校验
    inline std::optional<std::string> checkField(const Json::Value& parent, 
                                                const std::string& field, 
                                                Json::ValueType expect_type) {
        if (!parent.isMember(field)) {
            return "缺失字段: " + field;
        }
        if (parent[field].type() != expect_type) {
            return "字段类型错误: " + field + " (期望类型ID: " + std::to_string(expect_type) + ")";
        }
        return std::nullopt;
    }

    // 数组长度校验
    inline std::optional<std::string> checkArray(const Json::Value& parent, 
                                                const std::string& field, 
                                                size_t expect_size) {
        auto err = checkField(parent, field, Json::arrayValue);
        if (err) return err;
        if (parent[field].size() != expect_size) {
            return "数组长度错误: " + field + " (期望: " + std::to_string(expect_size) + 
                   ", 实际: " + std::to_string(parent[field].size()) + ")";
        }
        return std::nullopt;
    }

    // 通用取值模板
    template<typename T>
    inline T get(const Json::Value& parent, const std::string& field) {
        if constexpr (std::is_same_v<T, std::string>) {
            return parent[field].asString();
        } else if constexpr (std::is_same_v<T, int>) {
            return parent[field].asInt();
        } else if constexpr (std::is_same_v<T, bool>) {
            return parent[field].asBool();
        } else {
            throw std::invalid_argument("不支持的取值类型");
        }
    }

    // 数组元素取值
    template<typename T>
    inline T getArrayElem(const Json::Value& arr, size_t idx) {
        if constexpr (std::is_same_v<T, int>) {
            return arr[idx].asInt();
        } else {
            throw std::invalid_argument("数组仅支持int类型");
        }
    }
}

// ===================== 核心解析函数实现 =====================
std::optional<std::string> readConfig(const std::string& json_path, 
                                     InitModelConfig& init_model, 
                                     InferenceConfig& inference) {
    using namespace JsonParser;

    // 1. 打开JSON文件
    std::ifstream json_file(json_path);
    if (!json_file.is_open()) {
        return "无法打开文件: " + json_path;
    }

    // 2. 解析JSON根节点
    Json::Value root;
    Json::Reader reader;
    if (!reader.parse(json_file, root, false)) {
        json_file.close();
        return "JSON语法错误: " + reader.getFormattedErrorMessages();
    }
    json_file.close();

    // 3. 解析init_model节点
    auto err = checkField(root, "init_model", Json::objectValue);
    if (err) return "init_model: " + *err;
    const Json::Value& init_node = root["init_model"];

    // 3.1 校验init_model字段
    if ((err = checkField(init_node, "alias", Json::stringValue))) return "init_model." + *err;
    if ((err = checkField(init_node, "class_cnt", Json::intValue))) return "init_model." + *err;
    if ((err = checkField(init_node, "model_type", Json::intValue))) return "init_model." + *err;
    if ((err = checkArray(init_node, "model_size", 2))) return "init_model." + *err;
    if ((err = checkField(init_node, "is_cuda", Json::booleanValue))) return "init_model." + *err;
    if ((err = checkField(init_node, "cuda_id", Json::intValue))) return "init_model." + *err;
    if ((err = checkField(init_node, "score_thres", Json::floatValue))) return "init_model." + *err;
    if ((err = checkField(init_node, "nms_thres", Json::floatValue))) return "init_model." + *err;
    if ((err = checkField(init_node, "mode_path", Json::stringValue))) return "init_model." + *err;
    if ((err = checkField(init_node, "classes_path", Json::stringValue))) return "init_model." + *err;

    // 3.2 赋值init_model
    init_model.alias = get<std::string>(init_node, "alias");
    init_model.class_cnt = get<int>(init_node, "class_cnt");
    init_model.model_type = get<int>(init_node, "model_type");
    init_model.model_size[0] = getArrayElem<int>(init_node["model_size"], 0);
    init_model.model_size[1] = getArrayElem<int>(init_node["model_size"], 1);
    init_model.is_cuda = get<bool>(init_node, "is_cuda");
    init_model.cuda_id = get<int>(init_node, "cuda_id");
    init_model.score_thres = get<float>(init_node, "score_thres");
    // 检查为0-1之间
    if (init_model.score_thres < 0 || init_model.score_thres > 1) {
        return "init_model.score_thres 必须在0-1之间";
    }
    init_model.nms_thres = get<float>(init_node, "nms_thres");
    // 检查为0-1之间
    if (init_model.nms_thres < 0 || init_model.nms_thres > 1) {
        return "init_model.nms_thres 必须在0-1之间";
    }
    init_model.mode_path = get<std::string>(init_node, "mode_path");
    init_model.classes_path = get<std::string>(init_node, "classes_path");

    // 4. 解析inference节点
    err = checkField(root, "inference", Json::objectValue);
    if (err) return "inference: " + *err;
    const Json::Value& infer_node = root["inference"];

    // 4.1 校验inference基础字段
    if ((err = checkField(infer_node, "is_crop", Json::booleanValue))) return "inference." + *err;
    if ((err = checkField(infer_node, "save_result", Json::booleanValue))) return "inference." + *err;
    if ((err = checkField(infer_node, "result_path", Json::stringValue))) return "inference." + *err;

    // 4.2 校验roi子节点
    if ((err = checkField(infer_node, "roi", Json::objectValue))) return "inference." + *err;
    const Json::Value& roi_node = infer_node["roi"];
    if ((err = checkField(roi_node, "x", Json::intValue))) return "inference.roi." + *err;
    if ((err = checkField(roi_node, "y", Json::intValue))) return "inference.roi." + *err;
    if ((err = checkField(roi_node, "w", Json::intValue))) return "inference.roi." + *err;
    if ((err = checkField(roi_node, "h", Json::intValue))) return "inference.roi." + *err;

    // 4.3 赋值inference
    inference.is_crop = get<bool>(infer_node, "is_crop");
    inference.save_result = get<bool>(infer_node, "save_result");
    inference.result_path = get<std::string>(infer_node, "result_path");
    inference.roi.x = get<int>(roi_node, "x");
    inference.roi.y = get<int>(roi_node, "y");
    inference.roi.w = get<int>(roi_node, "w");
    inference.roi.h = get<int>(roi_node, "h");

    return std::nullopt;
}

// ===================== 打印配置函数实现 =====================
void printConfig(const InitModelConfig& init_model, const InferenceConfig& inference) {
    std::cout << "\n===== InitModelConfig =====" << std::endl;
    std::cout << "alias: " << init_model.alias << "\nclass_cnt: " << init_model.class_cnt
              << "\nmodel_size: [" << init_model.model_size[0] << ", " << init_model.model_size[1] << "]"
              << "\nis_cuda: " << std::boolalpha << init_model.is_cuda << "\ncuda_id: " << init_model.cuda_id
              << "\nmode_path: " << init_model.mode_path << "\nclasses_path: " << init_model.classes_path << std::endl;

    std::cout << "\n===== InferenceConfig =====" << std::endl;
    std::cout << "is_crop: " << std::boolalpha << inference.is_crop
              << "\nroi: x=" << inference.roi.x << ", y=" << inference.roi.y 
              << ", w=" << inference.roi.w << ", h=" << inference.roi.h
              << "\nsave_result: " << inference.save_result 
              << "\nresult_path: " << inference.result_path << std::endl;
}