import { useState } from "react";
import classNames from "classnames";
import Login from "../login/login";

function Settings() {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button
        className="bg-gray-800/70 rounded-md text-xs text-gray-300 hover:brightness-125 px-3 py-1.5 font-medium cursor-pointer"
        onClick={() => setOpen(!open)}
      >
        Settings
      </button>
      <div
        className={classNames(
          "h-screen w-screen bg-black/20 fixed left-0 top-0 z-10",
          {
            "opacity-0 pointer-events-none": !open,
          }
        )}
        onClick={() => setOpen(false)}
      ></div>
      <div
        className={classNames(
          "absolute top-[calc(100%+8px)] right-0 z-10 w-80 bg-white border border-gray-200 rounded-lg shadow-lg transition-all duration-75 overflow-hidden",
          {
            "opacity-0 pointer-events-none": !open,
          }
        )}
      >
        <Login />
      </div>
    </div>
  );
}
export default Settings;
