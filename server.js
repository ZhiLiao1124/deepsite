import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import cookieParser from "cookie-parser";
import { createRepo, uploadFiles, whoAmI } from "@huggingface/hub";
import { InferenceClient } from "@huggingface/inference";
import bodyParser from "body-parser";
import OpenAI from "openai";

import checkUser from "./middlewares/checkUser.js";

// Load environment variables from .env file
dotenv.config();

const app = express();

const ipAddresses = new Map();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.APP_PORT || 3000;
const REDIRECT_URI =
  process.env.REDIRECT_URI || `http://localhost:${PORT}/auth/login`;
const MODEL_ID = "deepseek-ai/DeepSeek-V3-0324";
const MAX_REQUESTS_PER_IP = 4;

app.use(cookieParser());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, "dist")));

app.get("/api/login", (_req, res) => {
  res.redirect(
    302,
    `https://huggingface.co/oauth/authorize?client_id=${process.env.OAUTH_CLIENT_ID}&redirect_uri=${REDIRECT_URI}&response_type=code&scope=openid%20profile%20write-repos%20manage-repos%20inference-api&prompt=consent&state=1234567890`
  );
});
app.get("/auth/login", async (req, res) => {
  const { code } = req.query;

  if (!code) {
    return res.redirect(302, "/");
  }
  const Authorization = `Basic ${Buffer.from(
    `${process.env.OAUTH_CLIENT_ID}:${process.env.OAUTH_CLIENT_SECRET}`
  ).toString("base64")}`;

  const request_auth = await fetch("https://huggingface.co/oauth/token", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      Authorization,
    },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      code: code,
      redirect_uri: REDIRECT_URI,
    }),
  });

  const response = await request_auth.json();

  if (!response.access_token) {
    return res.redirect(302, "/");
  }

  res.cookie("hf_token", response.access_token, {
    httpOnly: false,
    secure: true,
    sameSite: "none",
    maxAge: 30 * 24 * 60 * 60 * 1000,
  });

  return res.redirect(302, "/");
});
app.get("/api/@me", checkUser, async (req, res) => {
  const { hf_token } = req.cookies;
  try {
    const request_user = await fetch("https://huggingface.co/oauth/userinfo", {
      headers: {
        Authorization: `Bearer ${hf_token}`,
      },
    });

    const user = await request_user.json();
    res.send(user);
  } catch (err) {
    res.clearCookie("hf_token");
    res.status(401).send({
      ok: false,
      message: err.message,
    });
  }
});

app.post("/api/deploy", checkUser, async (req, res) => {
  const { html, title, path } = req.body;
  if (!html || !title) {
    return res.status(400).send({
      ok: false,
      message: "Missing required fields",
    });
  }

  let newHtml = html;

  if (!path) {
    newHtml = html.replace(
      /<\/body>/,
      `<p style="border-radius: 8px; text-align: center; font-size: 12px; color: #fff; margin-top: 16px;position: fixed; left: 8px; bottom: 8px; z-index: 10; background: rgba(0, 0, 0, 0.8); padding: 4px 8px;">Made with <a href="https://enzostvs-deepsite.hf.space" style="color: #fff;" target="_blank" >DeepSite</a> <img src="https://enzostvs-deepsite.hf.space/logo.svg" alt="DeepSite Logo" style="width: 16px; height: 16px; vertical-align: middle;"></p></body>`
    );
  }

  const file = new Blob([newHtml], { type: "text/html" });
  file.name = "index.html"; // Add name property to the Blob

  const { hf_token } = req.cookies;
  try {
    const repo = {
      type: "space",
      name: path ?? "",
    };

    let readme;

    if (!path || path === "") {
      const { name: username } = await whoAmI({ accessToken: hf_token });
      const newTitle = title
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .split("-")
        .filter(Boolean)
        .join("-")
        .slice(0, 96);

      const repoId = `${username}/${newTitle}`;
      repo.name = repoId;
      await createRepo({
        repo,
        accessToken: hf_token,
      });
      readme = `---
title: ${newTitle}
emoji: 🐳
colorFrom: blue
colorTo: blue
sdk: static
pinned: false
tags:
  - deepsite
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference`;
    }

    const files = [file];
    if (readme) {
      const readmeFile = new Blob([readme], { type: "text/markdown" });
      readmeFile.name = "README.md"; // Add name property to the Blob
      files.push(readmeFile);
    }
    await uploadFiles({
      repo,
      files,
      accessToken: hf_token,
    });
    return res.status(200).send({ ok: true, path: repo.name });
  } catch (err) {
    return res.status(500).send({
      ok: false,
      message: err.message,
    });
  }
});

app.post("/api/ask-ai", async (req, res) => {
  const { prompt, html, previousPrompt } = req.body;
  if (!prompt) {
    return res.status(400).send({
      ok: false,
      message: "Missing required fields",
    });
  }

  res.setHeader("Content-Type", "text/plain");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  // 使用多个 OpenRouter API Key 自动切换
  const openRouterKeys = [
    process.env.OPENROUTER_API_KEY_1,
    process.env.OPENROUTER_API_KEY_2,
    process.env.OPENROUTER_API_KEY_3,
    process.env.OPENROUTER_API_KEY_4,
  ].filter(Boolean);

  let apiKeyIndex = 0;

  async function createOpenAIClient() {
    return new OpenAI({
      baseURL: "https://openrouter.ai/api/v1",
      apiKey: openRouterKeys[apiKeyIndex],
    });
  }

  async function getWorkingClient() {
    for (let i = 0; i < openRouterKeys.length; i++) {
      apiKeyIndex = i;
      try {
        const client = await createOpenAIClient();
        // 先做一次简单调用确保这个 key 可用
        await client.models.list();
        return client;
      } catch (e) {
        console.warn(`API key ${i + 1} failed, trying next...`);
        continue;
      }
    }
    throw new Error("No available OpenRouter API key.");
  }

  let completeResponse = "";

  try {
    const openai = await getWorkingClient();

    const stream = await openai.chat.completions.create({
      model: "deepseek/deepseek-v3-base:free", // 强制使用免费模型
      messages: [
        {
          role: "system",
          content:
            "ONLY USE HTML, CSS AND JAVASCRIPT. If you want to use ICON make sure to import the library first. Try to create the best UI possible by using only HTML, CSS and JAVASCRIPT. Also, try to ellaborate as much as you can, to create something unique. ALWAYS GIVE THE RESPONSE INTO A SINGLE HTML FILE",
        },
        ...(previousPrompt
          ? [{ role: "user", content: previousPrompt }]
          : []),
        ...(html
          ? [{ role: "assistant", content: `The current code is: ${html}.` }]
          : []),
        { role: "user", content: prompt },
      ],
      stream: true,
      extra_headers: {
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "DeepSite-Local",
      },
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || "";
      res.write(content);
      completeResponse += content;

      if (completeResponse.includes("</html>")) {
        break;
      }
    }

    res.end();
  } catch (error) {
    console.error("Error:", error);
    if (!res.headersSent) {
      res.status(500).send({
        ok: false,
        message: `You probably reached the MAX_TOKENS limit, context is too long. You can start a new conversation by refreshing the page.`,
      });
    } else {
      res.end();
    }
  }
});

app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
